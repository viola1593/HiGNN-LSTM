import torch
from torch import nn
import torch_geometric as pyg
from .mlp import MLP

'''Adapted from: https://github.com/corentin-dfg/graph4sen2dwater/blob/main/models/graphcast/mesh2grid.py'''
    
class Mesh2GridGNN(nn.Module):
    def __init__(self, grid_node_dim, mesh_node_dim, edge_dim, hid_dim, timestep_dim=None):
        super().__init__()
        self.edge_mlp = MLP(edge_dim+mesh_node_dim+grid_node_dim, hid_dim, hid_dim)
        self.grid_mlp = MLP(grid_node_dim+hid_dim, hid_dim, hid_dim)
        
        # FiLM adapters for timestep conditioning
        if timestep_dim is not None:
            self.edge_film = nn.Linear(timestep_dim, 2 * (edge_dim + mesh_node_dim + grid_node_dim))  # scale + shift
            self.grid_film = nn.Linear(timestep_dim, 2 * (grid_node_dim + hid_dim))  # scale + shift
        else:
            self.edge_film = None
            self.grid_film = None

    def forward(self, data, gamma_edge=None, beta_edge=None, gamma_node=None, beta_node=None):
        source_node = data['mesh', 'to', 'grid'].edge_index[0]
        target_node = data['mesh', 'to', 'grid'].edge_index[1]
        # Compute the new edge features and node features
        edge_input = torch.cat((data['mesh','to','grid'].edge_attr,data['mesh'].x[source_node],data['grid'].x[target_node]), dim=-1)
        
        # FiLM modulation for edge MLP input
        if gamma_edge is not None:
            edge_input = gamma_edge * edge_input + beta_edge
        
        edge_new = self.edge_mlp(edge_input)
        
        grid_input = torch.cat((data['grid'].x, pyg.utils.scatter(edge_new, target_node, dim=0, reduce='sum', dim_size=data['grid'].x.shape[0])), dim=-1)
        
        # FiLM modulation for grid MLP input
        if gamma_node is not None:
            grid_input = gamma_node * grid_input + beta_node
        
        grid_new = self.grid_mlp(grid_input)

        data['grid'].x = data['grid'].x + grid_new
        return data
    