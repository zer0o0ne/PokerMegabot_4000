import torch
from torch import nn
import numpy as np
from modules import *
from agents import *

class SimpleBrain:
    def __init__(self, num_agents, loss, agent_type, agent_args = {}):
        self.num_agents = num_agents
        self.indicies = list(range(num_agents))
        self.agents = [agent_type(i, **agent_args) for i in range(self.num_agents)]
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
            start = [agent(module, start) for agent in self.agents]
        
        return start[self.players[position]]

    def train(self, position, actions, reward):
        loss = self.loss(actions, reward)
        loss.backward(inputs = list(self.agents[self.players[position]].parameters()))
        self.optimizers[self.players[position]].step()
        return loss.item()