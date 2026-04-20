from torch import nn
from .mlp import MLP

'''Adapted from: https://github.com/corentin-dfg/graph4sen2dwater/blob/main/models/graphcast/embedder_original_graphcast.py'''

class Embedder(nn.Module):
    def __init__(self, grid_node_dim, ocean_node_dim, mesh_node_dim, edge_dim, hid_dim):
        super().__init__()
        self.grid_node_mlp = MLP(grid_node_dim, hid_dim, hid_dim)
        self.ocean_node_mlp = MLP(ocean_node_dim, hid_dim, hid_dim)
        self.mesh_node_mlp = MLP(mesh_node_dim, hid_dim, hid_dim)

        self.edge_mlp = MLP(edge_dim, hid_dim, hid_dim)


    def forward(self, data):
        
        data['grid'].x = self.grid_node_mlp(data['grid'].x)
        data['ocean'].x = self.ocean_node_mlp(data['ocean'].x)
        data['mesh'].x = self.mesh_node_mlp(data['mesh'].x)
        data['grid', 'to', 'mesh'].edge_attr = self.edge_mlp(data['grid', 'to', 'mesh'].edge_attr.reshape(-1, 1))
        data['ocean', 'to', 'mesh'].edge_attr = self.edge_mlp(data['ocean', 'to', 'mesh'].edge_attr.reshape(-1, 1))
        data['mesh', 'to', 'mesh'].edge_attr = self.edge_mlp(data['mesh', 'to', 'mesh'].edge_attr.reshape(-1, 1))
        data['mesh', 'to', 'grid'].edge_attr = self.edge_mlp(data['mesh', 'to', 'grid'].edge_attr.reshape(-1, 1))
        return data