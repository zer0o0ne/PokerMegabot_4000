from torch import nn
from player.modules import *

class SimpleAgent(nn.Module):
    def __init__(self, number, start_type, start_args = {}, module_args = {}, modules = {}):
        super().__init__()
        self.number = number
        self.start = start_type(**start_args)
        self.Modules = nn.ModuleDict({name: module(**module_args[name]) for name, module in modules.items()})
        self.module_names = [name for name in modules]

    def forward(self, module, x):
        return self.Modules[module](x[self.number], x[:self.number] + x[self.number + 1:])

    def get_modules(self):
        return self.module_names