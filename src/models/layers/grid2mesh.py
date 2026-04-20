import torch
from torch import nn
import torch_geometric as pyg
from .mlp import MLP

'''Adapted from: https://github.com/corentin-dfg/graph4sen2dwater/blob/main/models/graphcast/grid2mesh.py'''
       
class Grid2MeshGNN(nn.Module):
    def __init__(self, grid_node_dim, mesh_node_dim, edge_dim, hid_dim):
        super().__init__()
        self.edge_mlp = MLP(grid_node_dim+mesh_node_dim, hid_dim, hid_dim)
        self.mesh_mlp = MLP(mesh_node_dim+hid_dim, hid_dim, hid_dim)
        self.grid_mlp = MLP(grid_node_dim, hid_dim, hid_dim)


    def forward(self, data):
        
        source_node = data['grid', 'to', 'mesh'].edge_index[0]
        target_node = data['grid', 'to', 'mesh'].edge_index[1]
        source_node_ocean = data['ocean', 'to', 'mesh'].edge_index[0]
        target_node_ocean = data['ocean', 'to', 'mesh'].edge_index[1]
        

        edge_new = self.edge_mlp(torch.cat((data['grid'].x[source_node], data['mesh'].x[target_node]), dim=-1))
        edge_ocean_new = self.edge_mlp(torch.cat((data['ocean'].x[source_node_ocean], data['mesh'].x[target_node_ocean]), dim=-1))

        agg_grid = pyg.utils.scatter(edge_new, target_node, dim=0, reduce='sum', dim_size=data['mesh'].x.shape[0])
        agg_ocean = pyg.utils.scatter(edge_ocean_new, target_node_ocean, dim=0, reduce='sum', dim_size=data['mesh'].x.shape[0])
        aggregated_features = agg_grid + agg_ocean

        mesh_new = self.mesh_mlp(torch.cat((data['mesh'].x, aggregated_features), dim=-1))

        grid_new = self.grid_mlp(data['grid'].x)
        ocean_new = self.grid_mlp(data['ocean'].x)


        data['grid'].x = data['grid'].x + grid_new
        data['mesh'].x = data['mesh'].x + mesh_new
        data['ocean'].x = data['ocean'].x + ocean_new
        data['grid', 'to', 'mesh'].edge_attr = data['grid', 'to', 'mesh'].edge_attr + edge_new
        data['ocean', 'to', 'mesh'].edge_attr = data['ocean', 'to', 'mesh'].edge_attr + edge_ocean_new
        return data
    
