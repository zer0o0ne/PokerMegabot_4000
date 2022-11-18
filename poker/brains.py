import torch
from torch import nn
import numpy as np
from modules import *
from agents import *
from collections import deque

class SimpleBrain:
    def __init__(self, num_agents, loss, memory, agent_type, emb_type, emb_args = {}, agent_args = {}):
        self.memory = memory
        self.num_agents = num_agents
        self.embedding = emb_type(**emb_args)
        self.indicies = list(range(num_agents))
        self.agents = [agent_type(i, **agent_args) for i in range(self.num_agents)]
        self.modules = self.agents[0].get_modules()
        self.loss = loss
        self.optimizers = [torch.optim.Adam(list(self.agents[i].parameters()) + 
                                            list(self.embedding.parameters())) for i in range(self.num_agents)]

    def sit(self, number, players = None):
        if players is None:
            self.players = np.random.choice(self.indicies, size = (number,), replace = False)
        else:
            self.players = players

    def step(self, position, env_state):
        players_state = self.memory.get_state(self.players[env_state["active_positions"]])
        env_state = {**players_state, **env_state}
        env_state = self.embedding.get_full_state(env_state)
        start = [agent.start(env_state) for agent in self.agents]
        for module in self.modules:
            start = [agent(module, start) for agent in self.agents]
        
        self.memory.archive(env_state["concatenation"], start[self.players[position]]["action"], self.players[position])
        return start[self.players[position]]

    def backward(self, position, actions, reward):
        reward["table"] = self.embedding.get_cards(reward["table"])
        loss = self.loss(actions, reward)
        loss.backward(inputs = list(self.agents[self.players[position]].parameters()))
        return loss.item()
    
    def optimize(self, position):
        self.optimizers[self.players[position]].step()
        self.optimizers[self.players[position]].zero_grad()


class NeuralHistoryCompressor(nn.Module):
    def __init__(self, num_agents, depth, lstm_params, train_freq = 5, eps = 0.001):
        super().__init__()
        self.stories = [deque()] * num_agents
        self.depth = depth
        self.train_freq = train_freq
        self.freqs = [0] * num_agents
        self.compressors = nn.ModuleList([nn.LSTM(**lstm_params[i]) for i in range(len(lstm_params))])
        self.optimizer = torch.optim.Adam(self.parameters())
        self.hn = [torch.randn(lstm_params[i]["num_layers"], lstm_params[i]["hidden_size"]) for i in range(len(lstm_params))]
        self.cn = [torch.randn(lstm_params[i]["num_layers"], lstm_params[i]["hidden_size"]) for i in range(len(lstm_params))]
        self.mse = nn.MSELoss()
        self.eps = eps

    def get_state(self, players):
        return {"players_state" : {player : list(self.stories[player]) for player in players} }

    def archive(self, env_state, action, player):
        predicted = False
        for level in range(self.depth):
            env_state, (self.hn[level], self.cn[level]) = self.compressors[level](env_state, (self.hn[level], self.cn[level]))
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