from env.dealers import SimpleDealer
from env.table import Table

from player.agents import SimpleAgent
from player.modules import *
from player.brains import SimpleBrain, NeuralHistoryCompressor

from utils import *

from time import time

path = "poker/base_configs.yaml"
configs = get_configs(path)
dealer = get_dealer(configs)
t = time()
dealer.game(configs["n_games"], configs["num_players"], configs["relocation_freq"], 
            configs["brain_train_freq"], configs["checkpoint_freq"], 
            configs["device"], configs["name"], with_human=configs["with_human"], reset_all=configs["reset_all"])
print(configs["n_games"], "games are finished in", time() - t, "seconds")