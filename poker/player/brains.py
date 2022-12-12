import torch
from torch import nn
import numpy as np
from player.modules import *
from player.agents import *
from collections import deque
from time import time

class SimpleBrain(nn.Module):
    def __init__(self, num_agents, loss, memory, agent_type, embedding, agent_args = {}):
        super().__init__()
        self.memory = memory
        self.num_agents = num_agents
        self.embedding = embedding
        self.indicies = list(range(num_agents))
        self.agents = nn.ModuleList([agent_type(i, **agent_args) for i in range(self.num_agents)])
        self.modules = self.agents[0].get_modules()
        self.loss = 0
        self.parameters = list(self.embedding.parameters())
        for i in range(num_agents):
            self.parameters += list(self.agents[i].parameters())
        self.optimizer = torch.optim.Adam(self.parameters)
        self.criterion = loss

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

    def save_loss(self, actions, reward):
        losses = []
        reward["table"] = self.embedding.get_cards(reward["table"])
        for position in range(self.num_players):
            reward["reward"] = reward["rewards"][position]
            loss = self.criterion(actions[position], reward)
            if loss is None: continue
            self.loss += loss
            losses.append(loss.item())
        return losses
    
    def optimize(self):
        self.loss.backward(inputs = self.parameters)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.loss = 0

    def init_history__(self, env_state, action, n_players):
        env_state = self.embedding.get_full_state(env_state)
        for position in range(n_players):
            if len(self.memory.stories[self.players[position]]) == 0:
                self.memory.archive(env_state, action, self.players[position])

    def set_device(self, device):
        self.to(device)
        self.memory.set_device(device)
        self.embedding.set_device(device)
        self.criterion.set_device(device)


#Support class
class NeuralHistoryCompressor(nn.Module):
    def __init__(self, num_agents, depth, memory_params, extractor_parameters, train_freq = 15, eps = 0.001):
        super().__init__()
        self.stories = [deque() for _ in range(num_agents)]
        self.depth, self.eps, self.train_freq, self.count, self.num_compressors = depth, eps, train_freq, 0, len(memory_params)
        self.compressors = nn.ModuleList([Transformer(**memory_params[i]) for i in range(len(memory_params))])
        self.mse = nn.MSELoss()
        self.extractor = Transformer(**extractor_parameters)
        self.numerator = nn.Embedding(num_agents, 2)
        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss, self.device = 0, "cpu"

    def get_state(self, players):
        players_state = [torch.cat([c.unsqueeze(0) for c in list(self.stories[player])]).to(self.device) for player in players]
        return {"players_state" : players_state}

    def set_device(self, device):
        self.device = device

    def archive(self, env_state, action, player):
        predicted = False
        action = action.unsqueeze(0)
        env_state = self.extractor(env_state["neuron_table_state"], env_state["neuron_now"])
        predict = env_state
        for level in range(self.num_compressors):
            predict, predicted = self.predict__(player, level, action, predict, predicted)
            if predicted: break
        if not predicted:
            self.save__(player, env_state, action)

        return env_state

    def predict__(self, player, level, action, predict, predicted):
        history = list(self.stories[player])
        if len(history) == 0: return predict, predicted
        predict = self.compressors[level](torch.cat(history).to(self.device), predict)
        divergence = self.mse(predict, action)
        if divergence.item() < self.eps:
            predicted = True
            return predict, predicted
        self.loss += divergence
        return predict, predicted

    def save__(self, player, env_state, action):
        self.count = (self.count + 1) % self.train_freq
        player_number = self.numerator(torch.tensor(player).to(self.device)).view((1, -1))
        self.stories[player].append(torch.cat([env_state, action, player_number], axis = 1).detach())
        if len(self.stories[player]) > self.depth:
            self.stories[player].popleft()
        if self.count == 0:
            self.loss.backward(inputs = list(self.parameters()))
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.loss = 0
    