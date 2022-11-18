import torch
from torch import nn
import numpy as np

class ConditionalExpectation_Loss:
    def __init__(self, function_args = [], functions = [], possible_events = []):
        self.Modules = nn.ModuleList([functions[i](**function_args[i]) for i in range(len(functions))])
        self.possible_events = possible_events

    def __call__(self, actions, reward):
        loss = 0
        for action in actions:
            action_exp = 0
            for step in range(4): # Sum around known events 
                with torch.no_grad():
                    self.Modules[step].set_weights(action["exp_f_weights"][step])
                action_exp += self.Modules[step](reward["table"][step])
            
            loss += torch.abs(action_exp - reward["reward"])
            probabilities = nn.Softmax(action["action"])
            loss += torch.log(probabilities[torch.argmax(probabilities)]) * (reward["reward"] - action_exp)

        return loss / len(actions)