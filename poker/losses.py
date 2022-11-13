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
            action_loss = 0
            for step in range(reward["steps"]): # Sum around known events 
                with torch.no_grad():
                    self.Modules[step].set_weights(action["exp_f_weights"][step])
                action_loss += self.Modules[step](reward["table"][step])

            loss += torch.abs(action_loss - reward["reward"])

        return loss / len(actions)