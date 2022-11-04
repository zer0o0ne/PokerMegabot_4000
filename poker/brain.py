import torch
from torch import nn
import numpy as np
from modules import *


class Agent(nn.Module):
    def __init__(self, number, emb_args = {}, start_args = {}, module_args = {}, modules = {}):
        super().__init__()
        self.number = number
        self.start = nn.Sequential(
            Embedding(**emb_args),
            MLP(**start_args)
        )
        self.Modules = nn.ModuleDict({name: module(**module_args[name]) for name, module in modules.items()})
        self.module_names = [name for name in modules]

    def forward(self, x):
        pass

    def get_modules(self):
        return self.module_names
    


class Brain:
    def __init__(self, num_agents, loss, agent_args = {}):
        self.num_agents = num_agents
        self.indicies = list(range(num_agents))
        self.agents = [Agent(i, **agent_args) for i in range(self.num_agents)]
        self.modules = self.agents[0].get_modules()
        self.loss = loss
        self.optimizers = [torch.optim.Adam(self.agents[i].parameters()) for i in range(self.num_agents)]

    def sit(self, number, players = None):
        if players is None:
            self.players = np.random.choice(self.indicies, size = (number,), replace = False)
        else:
            self.players = players

    def step(self, position, env_state):
        start = [agent.start(env_state) for agent in self.agents]
        for module in self.modules:
            start = [agent.Modules[module](start) for agent in self.agents]
        
        return start[self.players[position]]

    def train(self, position, action, reward):
        loss = self.loss(action, reward)
        loss.backward(inputs = list(self.agents[self.players[position]].parameters()))
        self.optimizers[self.players[position]].step()
        return loss.item()