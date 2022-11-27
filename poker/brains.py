import torch
from torch import nn
import numpy as np
from modules import *
from agents import *
from collections import deque

class SimpleBrain:
    def __init__(self, num_agents, loss, memory, agent_type, embedding, agent_args = {}):
        self.memory = memory
        self.num_agents = num_agents
        self.embedding = embedding
        self.indicies = list(range(num_agents))
        self.agents = [agent_type(i, **agent_args) for i in range(self.num_agents)]
        self.modules = self.agents[0].get_modules()
        self.loss = loss
        self.optimizers = [torch.optim.Adam(self.agents[i].parameters()) for i in range(self.num_agents)]
        self.emb_opt = torch.optim.Adam(self.embedding.parameters())

    def sit(self, number, players = None):
        self.num_players = number
        if players is None:
            self.players = np.random.choice(self.indicies, size = (number,), replace = False)
        else:
            self.players = players

    def rotate(self):
        idx = [len(self.players) - 1] + list(range(len(self.players) - 1))
        self.players = self.players[idx]

    def step(self, position, env_state):
        players_state = self.memory.get_state(self.players[env_state["active_positions"]])
        env_state = {**players_state, **env_state}
        env_state = self.embedding.get_full_state(env_state)
        start = [agent.start(env_state) for agent in self.agents]
        for module in self.modules[:-1]:
            start = [agent(module, start) for agent in self.agents]
        
        finish = self.agents[self.players[position]](self.modules[-1], start)
        self.memory.archive(env_state, finish["action"], self.players[position])
        return finish

    def backward(self, actions, reward):
        losses = []
        reward["table"] = self.embedding.get_cards(reward["table"])
        for position in range(self.num_players):
            reward["reward"] = reward["rewards"][position]
            loss = self.loss(actions[position], reward)
            if loss is None: continue
            loss.backward(inputs = list(self.agents[self.players[position]].parameters()))
            losses.append(loss.item())
        return losses
    
    def optimize(self):
        self.emb_opt.step()
        self.emb_opt.zero_grad()
        for position in range(self.num_players):
            self.optimizers[self.players[position]].step()
            self.optimizers[self.players[position]].zero_grad()


#Support class
class NeuralHistoryCompressor(nn.Module):
    def __init__(self, num_agents, depth, lstm_params, extractor_parameters, train_freq = 5, eps = 0.001):
        super().__init__()
        self.stories = [deque()] * num_agents
        self.depth, self.train_freq, self.eps, self.num_compressors = depth, train_freq, eps, len(lstm_params)
        self.freqs = [0] * num_agents
        self.compressors = nn.ModuleList([nn.LSTM(**lstm_params[i]) for i in range(len(lstm_params))])
        self.optimizer = torch.optim.Adam(self.parameters())
        self.hn = [[torch.randn(lstm_params[i]["num_layers"], lstm_params[i]["hidden_size"]) for i in range(self.num_compressors)]] * num_agents
        self.cn = [[torch.randn(lstm_params[i]["num_layers"], lstm_params[i]["hidden_size"]) for i in range(self.num_compressors)]] * num_agents
        self.mse = nn.MSELoss()
        self.extractor = nn.Transformer(**extractor_parameters)

    def get_state(self, players):
        players_state = [torch.cat([c.unsqueeze(0) for c in list(self.stories[player])]) for player in players]
        return {"players_state" : players_state}

    def archive(self, env_state, action, player):
        predicted = False
        env_state = self.extractor(env_state["neuron_table_state"], env_state["neuron_now"])
        for level in range(self.num_compressors):
            env_state, (self.hn[player][level], self.cn[player][level]) = self.compressors[level](env_state, (self.hn[player][level], self.cn[player][level]))
            divergence = self.mse(env_state, action)
            if divergence.item() < self.eps:
                predicted = True
                break
            divergence.backward()

        if not predicted:
            self.stories[player].append((env_state, action))
            if len(self.stories[player]) > self.depth:
                self.stories[player].popleft()
            self.freqs[player] = (self.freqs[player] + 1) % self.train_freq
            if self.freqs[player] == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        return env_state