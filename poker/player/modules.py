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
        self.device = "cpu"
        self.coefs = torch.randn((dim_in, 2 * depth)).to(self.device)
        self.depth = depth

    def set_weights(self, w):
        self.coefs = w.to(self.device)

    def set_device(self, device):
        self.device = device

    def forward(self, x):
        x = x.flatten()
        sinuses = [torch.sin(x * 6.28 * k).unsqueeze(-1) for k in range(1, self.depth + 1)]
        cosinuses = [torch.cos(x * 6.28 * k).unsqueeze(-1) for k in range(1, self.depth + 1)]
        decomposition = torch.cat(sinuses + cosinuses, axis = -1)
        return (decomposition * self.coefs).sum()


class SimpleEmbedding(nn.Module):
    def __init__(self, num_cards, hidden_dim, num_players, start_credits):
        super().__init__()
        self.emb = nn.Embedding(num_cards, hidden_dim)
        self.num_players = num_players
        self.start_credits = start_credits
        self.device = "cpu"

    def cards__(self, cards):
        return torch.cat([self.emb(torch.tensor(card).to(self.device) + 1).unsqueeze(0) for card in cards], axis = 0)

    def get_cards(self, list_of_steps):
        return [self.cards__(cards) for cards in list_of_steps]

    def set_device(self, device):
        self.device = device

    def get_full_state(self, env_state):
        table_state = []
        for step in env_state["table_state"]:
            pos = torch.tensor(step["pos"], dtype = torch.float).to(self.device).unsqueeze(0) / self.num_players
            pot = torch.tensor(step["pot"], dtype = torch.float).to(self.device).unsqueeze(0) / self.start_credits
            table = self.cards__(step["table"]).flatten()
            table_state.append(torch.cat([pos, pot, step["action"], table]).unsqueeze(0))
        table_state = torch.cat(table_state)
        
        pos = torch.tensor(env_state["now"]["pos"], dtype = torch.float).to(self.device).unsqueeze(0) / self.num_players
        pot = torch.tensor(env_state["now"]["pot"], dtype = torch.float).to(self.device).unsqueeze(0) / self.start_credits
        bank = torch.tensor(env_state["now"]["bank"], dtype = torch.float).to(self.device).unsqueeze(0) / self.start_credits
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
        x_inn = self.inner_input(x_inner)
        x_outer = torch.cat([x.unsqueeze(0) for x in x_outer]).detach()
        x_outer = self.agents_weights @ x_outer
        x_outer = self.outer_input(x_outer)
        x = torch.cat([x_inn, x_outer])
        for layer in self.process:
            x = layer(x)
        return {"last_output": self.output(x), "self_transformer_output": x_inner}


class FourierOutput(Numbered_MLP):
    def __init__(self, dim_in, hidden_dim, dim_encoder_output, depth, num_agents, shapes):
        dim_out = np.sum([np.prod(shape) for shape in shapes])
        super().__init__(dim_in + dim_encoder_output, hidden_dim, dim_out, depth, num_agents)
        self.shapes = shapes

    def forward(self, x_inner, x_outer):
        x_i = self.inner_input(torch.cat([x_inner["last_output"], x_inner["self_transformer_output"]], axis = -1))
        x_outer = torch.cat([torch.cat([x["last_output"], x["self_transformer_output"]], axis = -1).unsqueeze(0) for x in x_outer]).detach()
        x_outer = self.agents_weights @ x_outer
        x_outer = self.outer_input(x_outer)
        x = torch.cat([x_i, x_outer])
        for layer in self.process:
            x = layer(x)
        x = self.output(x).flatten()
        exp_f_weights, i = [], 0
        for shape in self.shapes:
            exp_f_weights.append(x[i : i + np.prod(shape)].view(shape))
            i += np.prod(shape)

        return {"action": x_inner["last_output"], "exp_f_weights": exp_f_weights}


class TransformersInput(nn.Module):
    def __init__(self, first_args, second_args):
        super().__init__()
        self.table_transformer = Transformer(**first_args)
        self.player_transformer = Transformer(**second_args)

    def forward(self, x):
        table_state = self.table_transformer(x["neuron_table_state"], x["neuron_now"])
        player_state = self.player_transformer(torch.cat(x["players_state"]).squeeze(1), table_state)
        return (torch.cat([table_state.flatten(), player_state.flatten()]))


class Transformer(nn.Module):
    def __init__(self, dim_source, dim_target, transformer_params, positional_encoding = False):
        super().__init__()
        self.source_projection = nn.Linear(dim_source, transformer_params["d_model"])
        self.target_projection = nn.Linear(dim_target, transformer_params["d_model"])
        self.transformer = nn.Transformer(**transformer_params)

    def forward(self, source, target):
        source = self.source_projection(source)
        target = self.target_projection(target)
        return self.transformer(source, target)