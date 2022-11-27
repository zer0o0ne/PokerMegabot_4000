import torch
from torch import nn
import numpy as np
from math import *

class ConditionalExpectation_Loss:
    def __init__(self, function_args = [], functions = []):
        self.Modules = nn.ModuleList([functions[i](**function_args[i]) for i in range(len(functions))])

    def __call__(self, actions, reward):
        loss = 0
        if len(actions) == 0:
            print("Warning! Some player didn`t any action!")
            return None
        for action in actions:
            action_exp = 0
            for step in range(3): # Sum around known events 
                with torch.no_grad():
                    self.Modules[step].set_weights(action["exp_f_weights"][step])
                action_exp += self.Modules[step](reward["table"][step])
            
            loss += torch.abs(action_exp - reward["reward"])
            probabilities = nn.Softmax(action["action"])
            loss += torch.log(probabilities[torch.argmax(probabilities)]) * (reward["reward"] - action_exp)

        return loss / len(actions)

#TODO
def get_configs():
    return 0
