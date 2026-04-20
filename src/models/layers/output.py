import torch
from torch import nn
from .mlp import MLP
'''Adapted from: https://github.com/corentin-dfg/graph4sen2dwater/blob/main/models/graphcast/output.py'''

class Outputter(nn.Module):
    def __init__(self, grid_node_dim, hid_dim, out_dim):
        super().__init__()
        if out_dim > 1:
            self.grid_node_mlp = MLP(grid_node_dim, hid_dim, out_dim, last=False)
        else:
            self.grid_node_mlp = MLP(grid_node_dim, hid_dim, out_dim, last=True)

    def forward(self, data):
        return self.grid_node_mlp(data['grid'].x)