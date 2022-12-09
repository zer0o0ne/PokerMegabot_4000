from env.dealers import SimpleDealer
from env.table import Table

from player.agents import SimpleAgent
from player.modules import *
from player.brains import SimpleBrain, NeuralHistoryCompressor

from utils import ConditionalExpectation_Loss

from time import time

num_players = 9
num_agents = 12
num_cards = 52 + 1
bins = 12
actions_dim = bins + 4
embedding_hidden_dim = 6
feedforward_dim = 24
hidden_size = 36
embedding_output_dim_table = 2 + 5 * embedding_hidden_dim + actions_dim
embedding_output_dim_now = 3 + 7 * embedding_hidden_dim
history_compressor_depth = 25
start_credits = 1000

memory_params = [
    {
        "dim_source" : hidden_size + actions_dim, 
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
        "dim_source" : hidden_size + actions_dim, 
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
            "dim_source" : hidden_size + actions_dim, 
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

table = Table(num_players, bins = bins)

loss = ConditionalExpectation_Loss(start_credits, function_args, functions)
memory = NeuralHistoryCompressor(num_agents, history_compressor_depth, memory_params, extractor_parameters, train_freq = 32)
embedding = SimpleEmbedding(num_cards, embedding_hidden_dim, num_players, start_credits)

brain = SimpleBrain(num_agents, loss, memory, SimpleAgent, embedding, agent_args)

dealer = SimpleDealer(table, brain)

n_games = 250

t = time()
dealer.game(n_games, num_players, 24, 3, device = "cuda")
print()
print()
print(n_games, "games are finished in time", time() - t, "seconds")