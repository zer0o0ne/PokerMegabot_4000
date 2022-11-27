import torch
from torch import nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out, depth):
        super().__init__()
        self.depth = depth
        self.input = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.PReLU()
        )
        self.process = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.PReLU()
            ) for i in range(depth)
        ])
        self.output = nn.Linear(hidden_dim, dim_out)

    def forward(self, x):
        x = self.input(x)
        for layer in self.process:
            x = layer(x)
        return self.output(x)


class Fourier(nn.Module):
    def __init__(self, dim_in, depth):
        super().__init__()
        self.weights = torch.randn((dim_in, 2 * depth))
        self.depth = depth

    def set_weights(self, w):
        self.weights = w

    def forward(self, x):
        x = x.flatten()
        sinuses = [torch.sin(x * 6.28 * k).unsqueeze(-1) for k in range(1, self.depth + 1)]
        cosinuses = [torch.cos(x * 6.28 * k).unsqueeze(-1) for k in range(1, self.depth + 1)]
        decomposition = torch.cat(sinuses + cosinuses, axis = -1)
        return (decomposition * self.weights).sum()


class SimpleEmbedding(nn.Module):
    def __init__(self, num_cards, hidden_dim, num_players):
        super().__init__()
        self.emb = nn.Embedding(num_cards, hidden_dim)
        self.num_players = num_players

    def cards__(self, cards):
        return torch.cat([self.emb(torch.tensor(card)).unsqueeze(0) for card in cards], axis = 0)

    def get_cards(self, list_of_steps):
        return [self.cards__(cards) for cards in list_of_steps]

    def get_full_state(self, env_state):
        table_state = []
        for step in env_state["table_state"]:
            pos = torch.tensor(step["pos"], dtype = torch.float).unsqueeze(0) / self.num_players
            pot = torch.tensor(step["pot"], dtype = torch.float).unsqueeze(0) / 100
            table = self.cards__(step["table"]).flatten()
            table_state.append(torch.cat([pos, pot, step["action"], table]).unsqueeze(0))
        table_state = torch.cat(table_state)
        
        pos = torch.tensor(env_state["now"]["pos"], dtype = torch.float).unsqueeze(0) / self.num_players
        pot = torch.tensor(env_state["now"]["pot"], dtype = torch.float).unsqueeze(0) / 100
        bank = torch.tensor(env_state["now"]["bank"], dtype = torch.float).unsqueeze(0) / 100
        hand = self.cards__(env_state["now"]["hand"]).flatten()
        table = self.cards__(env_state["now"]["table"]).flatten()
        now = torch.cat([pos, pot, bank, hand, table], axis = 0).unsqueeze(0)

        env_state["neuron_table_state"] = table_state
        env_state["neuron_now"] = now
        return env_state


class Numbered_MLP(MLP):
    def __init__(self, dim_in, hidden_dim, dim_out, depth, num_agents):
        super().__init__(dim_in, hidden_dim, dim_out, depth)
        self.agents_weights = nn.Parameter(torch.randn(num_agents - 1,))
        self.inner_input = nn.Sequential(
            nn.Linear(dim_in, hidden_dim // 2),
            nn.PReLU()
        )
        self.outer_input = nn.Sequential(
            nn.Linear(dim_in, hidden_dim // 2 + hidden_dim % 2),
            nn.PReLU()
        )

    def forward(self, x_inner, x_outer):
        x_inner = self.inner_input(x_inner)
        x_outer = torch.cat([x.unsqueeze(0) for x in x_outer])
        x_outer = self.agents_weights @ x_outer
        x_outer = self.outer_input(x_outer)
        x = torch.cat([x_inner, x_outer])
        for layer in self.process:
            x = layer(x)
        return self.output(x)


class FourierOutput(Numbered_MLP):
    def __init__(self, dim_in, hidden_dim, depth, num_agents, shape):
        dim_out = np.prod(shape)
        super().__init__(dim_in, hidden_dim, dim_out, depth, num_agents)
        self.shape = shape

    def forward(self, x_inner, x_outer):
        x_i = self.inner_input(x_inner)
        x_outer = torch.cat([x.unsqueeze(0) for x in x_outer])
        x_outer = self.agents_weights @ x_outer
        x_outer = self.outer_input(x_outer)
        x = torch.cat([x_i, x_outer])
        for layer in self.process:
            x = layer(x)
        return {"action": x_inner, "exp_f_weights": self.output(x).view(self.shape)}


class TransformersInput(nn.Module):
    def __init__(self, first_args, second_args):
        super().__init__()
        self.table_transformer = nn.Transformer(**first_args)
        self.player_transformer = nn.Transformer(**second_args)

    def forward(self, x):
        table_state = self.table_transformer(x["neuron_table_state"], x["neuron_now"].unsqueeze(0))
        player_state = self.player_transformer(torch.cat(x["players_state"]), table_state)
        return (torch.cat([table_state.flatten(), player_state.flatten()]))