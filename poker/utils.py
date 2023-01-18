import torch
from torch import nn
import numpy as np
from math import *
import yaml

from env.dealers import SimpleDealer
from env.table import Table

from player.agents import SimpleAgent
from player.modules import *
from player.brains import SimpleBrain, NeuralHistoryCompressor

class ConditionalExpectation_Loss:
    def __init__(self, start_credits, function_args = [], functions = [], fold_discount = 0.03):
        self.start_credits = start_credits
        self.fold_discount = fold_discount
        self.Modules = nn.ModuleList([functions[i](**function_args[i]) for i in range(len(functions))])
        self.device = "cpu"

    def set_device(self, device):
        self.device = device
        for i in range(len(self.Modules)):
            self.Modules[i].set_device(device)

    def __call__(self, actions, reward):
        loss, pred_loss = 0, 0
        if len(actions) == 0:
            return None, None
        for action in actions:
            action_exp = 0
            for step in range(3): # Sum around known events 
                with torch.no_grad():
                    self.Modules[step].set_weights(action["exp_f_weights"][step])
                action_exp += self.Modules[step](reward["table"][step])
            
            pred_loss += torch.abs(action_exp - reward["reward"] / self.start_credits)
            probabilities = nn.Softmax(dim = -1)(action["action"])
            loss += torch.log(probabilities[torch.argmax(probabilities)]) * (action_exp - reward["reward"] / self.start_credits)
            loss += torch.log(probabilities[0]) * self.fold_discount

        return loss, pred_loss

def get_loss(config):
    return ConditionalExpectation_Loss(config["start_credits"], config["function_args"], config["functions"], config["fold_discount"])

def get_memory(config):
    return NeuralHistoryCompressor(config["num_agents"], config["history_compressor_depth"], config["memory_params"], config["extractor_parameters"], 
                                    config["history_train_freq"])

def get_embedding(config):
    return SimpleEmbedding(config["num_cards"], config["embedding_hidden_dim"], config["num_players"], config["start_credits"])

def get_table(config):
    return Table(config["num_players"], config["bins"], big_blind = config["big_blind"], small_blind = config["small_blind"], 
                start_credits = config["start_credits"], max_bet = config["max_bet"])

def get_brain(config):
    loss = get_loss(config)
    memory = get_memory(config)
    embedding = get_embedding(config)
    return SimpleBrain(config["num_agents"], loss, memory, SimpleAgent, embedding, config["agent_args"])

def get_dealer(config):
    brain = get_brain(config)
    table = get_table(config)
    return SimpleDealer(table, brain)

def get_configs(filename):

    with open(filename) as file:
        raw_configs = yaml.load(file, Loader=yaml.FullLoader)

    num_players = raw_configs["num_players"]
    num_agents = raw_configs["num_agents"]
    num_cards = raw_configs["num_cards"] + 1
    bins = raw_configs["bins"]
    max_bet = raw_configs["max_bet"]
    with_human = raw_configs["with_human"]
    reset_all = raw_configs["reset_all"]
    actions_dim = bins + 3
    checkpoint_freq = raw_configs["checkpoint_freq"]
    name = raw_configs["name"]
    embedding_hidden_dim = raw_configs["embedding_hidden_dim"]
    feedforward_dim = raw_configs["feedforward_dim"]
    hidden_size = raw_configs["hidden_size"]
    embedding_output_dim_table = 2 + 5 * embedding_hidden_dim + actions_dim
    embedding_output_dim_now = 3 + 7 * embedding_hidden_dim
    history_compressor_depth = raw_configs["history_compressor_depth"]
    start_credits = raw_configs["start_credits"]
    n_games = raw_configs["n_games"]
    history_train_freq = raw_configs["history_train_freq"]
    brain_train_freq = raw_configs["brain_train_freq"]
    relocation_freq = raw_configs["relocation_freq"]
    device = raw_configs["device"]
    big_blind = raw_configs["big_blind"]
    small_blind = raw_configs["small_blind"]
    fold_discount = raw_configs["fold_discount"]

    memory_params = [
        {
            "dim_source" : hidden_size + actions_dim + 2, 
            "dim_target" : hidden_size,
            "transformer_params": {
                "d_model": actions_dim, 
                "nhead": 4, 
                "num_encoder_layers": 2, 
                "num_decoder_layers": 2, 
                "dim_feedforward": feedforward_dim
            }
        },

        {
            "dim_source" : hidden_size + actions_dim + 2, 
            "dim_target" : actions_dim,
            "transformer_params": {
                "d_model": actions_dim, 
                "nhead": 4, 
                "num_encoder_layers": 2, 
                "num_decoder_layers": 2, 
                "dim_feedforward": feedforward_dim
            }
        },
    ]

    extractor_parameters = {
        "dim_source" : embedding_output_dim_table, 
        "dim_target" : embedding_output_dim_now,
        "transformer_params": {
            "d_model": hidden_size, 
            "nhead": 4, 
            "num_encoder_layers": 2, 
            "num_decoder_layers": 2, 
            "dim_feedforward": feedforward_dim
        }
    }

    functions = [Fourier, Fourier, Fourier]
    function_args = [
        {"dim_in": embedding_hidden_dim * 3, "depth": 15},
        {"dim_in": embedding_hidden_dim, "depth": 15},
        {"dim_in": embedding_hidden_dim, "depth": 15},
    ]

    agent_args = {
        "start_type": TransformersInput,
        "start_args" : {
            "first_args" : {
                "dim_source" : embedding_output_dim_table, 
                "dim_target" : embedding_output_dim_now,
                "transformer_params": {
                    "d_model": hidden_size, 
                    "nhead": 4, 
                    "num_encoder_layers": 2, 
                    "num_decoder_layers": 2, 
                    "dim_feedforward": feedforward_dim
                }
            }, 
            "second_args" : {
                "dim_source" : hidden_size + actions_dim + 2, 
                "dim_target" : hidden_size,
                "transformer_params": {
                    "d_model": hidden_size, 
                    "nhead": 4, 
                    "num_encoder_layers": 2, 
                    "num_decoder_layers": 2, 
                    "dim_feedforward": feedforward_dim
                }
            }
        },

        "module_args" : {
            "First_Numbered_MLP" : {
                "dim_in" : hidden_size * 2, 
                "hidden_dim" : hidden_size,
                "dim_out" : actions_dim, 
                "depth" : 3, 
                "num_agents": num_agents
            },
            
            "FourierOutput" : {
                "dim_in" : actions_dim, 
                "dim_encoder_output": hidden_size * 2,
                "hidden_dim" : hidden_size,
                "shapes" : [[embedding_hidden_dim * 3, 30], [embedding_hidden_dim, 30], [embedding_hidden_dim, 30]], 
                "depth" : 3,
                "num_agents": num_agents
            }
        },

        "modules" : {
            "First_Numbered_MLP" : Numbered_MLP,
            "FourierOutput" : FourierOutput
        }
    }

    return {
        "memory_params": memory_params, "extractor_parameters": extractor_parameters, "functions": functions, "function_args": function_args,
        "agent_args": agent_args, "num_players": num_players, "bins": bins, "start_credits": start_credits, "num_agents": num_agents,
        "history_compressor_depth": history_compressor_depth, "history_train_freq": history_train_freq, "num_cards": num_cards, 
        "embedding_hidden_dim": embedding_hidden_dim, "device": device, "n_games": n_games, "brain_train_freq": brain_train_freq, 
        "relocation_freq": relocation_freq, "checkpoint_freq": checkpoint_freq, "name": name, "small_blind": small_blind, "big_blind": big_blind,
        "fold_discount": fold_discount, "max_bet": max_bet, "with_human": with_human, "reset_all": reset_all
    }

