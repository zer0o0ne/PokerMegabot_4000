import torch
import sys
import logging
import pickle
from datetime import datetime
from copy import deepcopy
from tqdm.auto import tqdm
import numpy as np
import os

class SimpleDealer:
    def __init__(self, table, brain):
        self.table = table
        self.brain = brain

    def game(self, n_games, n_players, relocation_freq = 180, training_freq = 45, checkpoint_freq = None, device = "cpu", name = "default", log_time = True):
        self.n_players = n_players
        self.device = device
        if checkpoint_freq is None:
            checkpoint_freq = relocation_freq * n_players
        self.init_names__(name, log_time)
        self.brain.sit(n_players)
        self.brain.set_device(device)
        self.init_history__()

        for game in tqdm(range(n_games), desc = "games"):
            if game % relocation_freq == relocation_freq - 1:
                self.brain.sit(n_players)
                self.init_history__()
                self.table.reset()

            self.game__(game)

            if game % training_freq == training_freq - 1:
                self.brain.optimize()   

            if  game % checkpoint_freq == checkpoint_freq - 1:
                self.checkpoint__(game)    
        logging.warning("Time of finish: " + datetime.now().strftime("%d/%m/%Y_%H:%M:%S"))   

    def game__(self, n):
        end = False
        self.table.start_table()
        table_state = self.table.get_state()
        self.log_start__(n)
        action = torch.zeros((self.table.bins + 3,)).to(self.device)
        env_state = self.prepare_state({"table_state": [], "now": 0}, table_state, action)
        env_state["now"] = table_state
        env_state["active_positions"] = table_state["active_positions"]
        actions = [[] for _ in range(self.n_players)] 
        step = 1
        while not end:
            action = self.brain.step(self.table.active_player, env_state)
            actions[self.table.active_player].append(action)
            env_state = self.prepare_state(env_state, table_state, action["action"])
            end, table_state_, bet = self.table.step(action["action"])
            self.log_step__(n, table_state, action["action"], bet)
            table_state = table_state_
            env_state["now"] = table_state
            env_state["active_positions"] = table_state["active_positions"]
            step += 1
        reward = self.table.get_reward()
        losses = self.brain.save_loss(actions, deepcopy(reward))
        self.log_game__(n, reward, losses)
        self.brain.rotate()
        self.table.rotate()
        for i in range(self.n_players):
            if self.table.credits[i] == 0:
                self.table.reset()

    def prepare_state(self, env_state, table_state, action):
        for_env = {
            "pos": table_state["pos"],
            "pot": table_state["pot"],
            "action": action,
            "table": table_state["table"]
        }
        env_state["table_state"].append(for_env)

        return env_state

    def init_history__(self):
        self.table.start_table(for_history = True)
        action = torch.zeros((self.table.bins + 3,)).to(self.device)
        table_state = self.table.get_state()
        env_state = self.prepare_state({"table_state": [], "now": 0}, table_state, action)
        env_state["now"] = table_state
        env_state["active_positions"] = table_state["active_positions"]
        self.brain.init_history__(env_state, action, self.n_players)

    def init_names__(self, name, log_time):
        self.logging_name = sys.path[0] + "/logging/" + name
        self.checkpoint_name = sys.path[0] + "/checkpoints/" + name
        if log_time:
            now = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
            self.logging_name += "_" + now
            self.checkpoint_name += "_" + now
        logging.basicConfig(filename = self.logging_name + ".txt", format='%(message)s')
        os.mkdir(self.checkpoint_name)

    def checkpoint__(self, n):
        file = self.checkpoint_name + "/" + str(n + 1) + ".pickle"
        with open(file, 'wb') as f:
            pickle.dump(self.brain, f)

    def log_start__(self, n):
        lengths = np.array([len(history) for history in self.brain.memory.stories])
        lengths = ", ".join(list(map(str, lengths[self.brain.players])))
        players = ", ".join(list(map(str, self.brain.players)))
        stacks = ", ".join(list(map(str, self.table.credits)))
        logging.warning('Game ' + str(n + 1) + ' starts')
        logging.warning('Active agents: ' + players)
        logging.warning('Their stacks: ' + stacks)
        logging.warning('Their histories lengths: ' + lengths)
        logging.warning(' ')

    def log_step__(self, n, table_state, action, bet):
        action = torch.argmax(action).item()
        bets = ""
        for pos in table_state["active_positions"]:
            bets += str(pos) + ": " + str(table_state["bets"][pos]) + ", "
        logging.warning('\tFlop: ' + self.decode_cards__(table_state["table"]))
        logging.warning('\tActive players bets: \t' + bets)
        logging.warning('\tPlayer ' + str(table_state["pos"]) + " has hand: " + self.decode_cards__(table_state["hand"]))
        if self.table.players_state[table_state["pos"]] == 2:
            logging.warning('\tPlayer goes all-in')
        elif action == 0:
            logging.warning('\tPlayer folds')
        elif action == 1:
            logging.warning('\tPlayer call or check')
        else: 
            logging.warning('\tPlayer bets ' + str(bet))
        logging.warning(' ')

    def log_game__(self, n, reward, losses):
        rewards = ", ".join(list(map(str, reward["rewards"])))
        losses = ", ".join(list(map(str, losses)))
        logging.warning('\tGame finished with rewards: ' + rewards)
        logging.warning('\tGame finished with losses: ' + losses)
        logging.warning(' ')
        logging.warning(' ')

    def decode_cards__(self, cards):
        suits = {
            0: "diamods",
            1: "hearts",
            2: "clubs",
            3: "spades"
        }
        prepare_cards = []
        for card in cards:
            if card != -1:
                suit = card % 4
                rank = card // 4 + 2
                if rank == 11: rank = "J"
                elif rank == 12: rank = "Q"
                elif rank == 13: rank = "K"
                elif rank == 14: rank = "A"
                prepare_cards.append(str(rank) + " of " + str(suits[suit])) 

        return ", ".join(prepare_cards)
