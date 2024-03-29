import torch
from torch import nn
import numpy as np
from math import *
from env.judger import Judger

class Table:
    def __init__(self, num_players, bins = 10, max_bet = 2, start_credits = 1000, big_blind = 10, small_blind = 5):
        self.num_players = num_players
        self.big_blind = big_blind
        self.small_blind = small_blind
        self.start_credits = start_credits
        self.credits = [start_credits] * num_players
        self.bins = bins
        self.max_bet = max_bet
        self.judger = Judger()

    def reset(self, position = None):
        if position is None:
            self.credits = [self.start_credits] * self.num_players
        else:
            self.credits[position] = self.start_credits

    def get_hand(self):
        pos = self.active_player
        return self.deck[5 + 2 * pos : 7 + 2 * pos]

    def rotate(self):
        idx = [self.num_players - 1] + list(range(self.num_players - 1))
        self.credits = list(np.array(self.credits)[idx])

    def start_table(self, for_history = False):
        self.players_state = np.ones((self.num_players,))
        self.active_player = 2
        self.several_all_in = False
        sb = min(self.small_blind, self.credits[0])
        bb = min(self.big_blind, self.credits[1])
        if not for_history:
            self.credits[0] -= sb
            self.credits[1] -= bb
            if self.credits[0] == 0: self.players_state[0] = 2
            if self.credits[1] == 0: self.players_state[1] = 2
        self.deck = np.random.permutation(52)
        self.pot = sb + bb
        self.high_bet = self.big_blind
        self.bets = np.zeros((self.num_players,))
        self.bets[0], self.bets[1] = sb, bb
        self.turn = 0

    def step(self, action):
        bet = 0
        if not self.several_all_in:
            action = torch.argmax(action).item()
            if action == 0:
                self.players_state[self.active_player] = -1

            if action == 1:
                bet = min(self.high_bet - self.bets[self.active_player], self.credits[self.active_player]) 
                self.pot += bet
                self.credits[self.active_player] -= bet
                self.bets[self.active_player] += bet
                self.players_state[self.active_player] = 0
                if self.credits[self.active_player] == 0: self.players_state[self.active_player] = 2

            if action > 1 and action < self.bins + 2:
                part_of_pot = (action - 1) * self.max_bet / self.bins
                bet = min(self.high_bet - self.bets[self.active_player] + part_of_pot * self.pot, self.credits[self.active_player])
                self.pot += bet
                self.credits[self.active_player] -= bet
                self.bets[self.active_player] += bet
                self.high_bet = max(self.high_bet, self.bets[self.active_player])
                self.players_state[self.active_player] = 0
                if self.credits[self.active_player] == 0: self.players_state[self.active_player] = 2

            if action == self.bins + 2:
                bet = self.credits[self.active_player]
                self.pot += bet
                self.credits[self.active_player] -= bet
                self.bets[self.active_player] += bet
                self.high_bet = max(self.high_bet, self.bets[self.active_player])
                self.players_state[self.active_player] = 2

        end = self.next_turn()
        return end, self.several_all_in, self.get_state(), bet

    def next_turn(self):
        active_players = self.players_state >= 0
        if active_players.sum() == 1:
            self.credits[np.argmax(active_players)] += self.pot
            return True

        for i in range(self.num_players):
            if self.bets[i] < self.high_bet and self.players_state[i] == 0:
                self.players_state[i] = 1

        waiting_players = self.players_state == 0
        moving_players = self.players_state == 1
        if moving_players.sum() == 0:
            if self.turn == 3:
                rewards = self.judger.get_reward(self.deck, self.players_state, self.bets)
                for i in range(self.num_players): 
                    self.credits[i] += rewards[i] + self.bets[i]
                return True
            else:
                self.turn += 1
                self.players_state[waiting_players] = 1
                self.active_player = np.argmax(waiting_players)
                if waiting_players.sum() == 0:
                    self.several_all_in = True
        
        else:
            self.active_player = (self.active_player + 1) % self.num_players
            while self.players_state[self.active_player] != 1:
                self.active_player = (self.active_player + 1) % self.num_players

        return False 

    def get_state(self):
        active_positions = np.arange(self.num_players)[self.players_state >= 0]
        pos = self.active_player
        pot = self.pot
        bank = self.credits[pos]
        hand = self.deck[5 + 2 * pos : 7 + 2 * pos]
        bets = np.copy(self.bets)
        if self.turn == 0: table = [-1] * 5
        if self.turn == 1: table = list(self.deck[:3]) + [-1] * 2
        if self.turn == 2: table = list(self.deck[:4]) + [-1]
        if self.turn == 3: table = list(self.deck[:5]) 
        return {"active_positions": active_positions, "pos": pos, "pot": pot, "bank": bank, "hand": hand, "table": table, "bets": bets}

    def get_reward(self):
        active_positions = np.arange(self.num_players)[self.players_state >= 0]
        hands = [self.deck[5 + 2 * pos : 7 + 2 * pos] for pos in np.arange(self.num_players)]
        table = [self.deck[:3], [self.deck[3]], [self.deck[4]]]
        rewards = self.judger.get_reward(self.deck, self.players_state, self.bets)
        return {"table": table, "rewards": rewards, "active_positions": active_positions, "hands": hands}


