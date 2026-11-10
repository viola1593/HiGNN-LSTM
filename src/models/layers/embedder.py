from torch import nn
from .mlp import MLP

'''Adapted from: https://github.com/corentin-dfg/graph4sen2dwater/blob/main/models/graphcast/embedder_original_graphcast.py'''

class Embedder(nn.Module):
    def __init__(self, grid_node_dim, ocean_node_dim, mesh_node_dim, edge_dim, hid_dim, use_ocean_nodes=True):
        super().__init__()
        self.use_ocean_nodes = use_ocean_nodes
        self.grid_node_mlp = MLP(grid_node_dim, hid_dim, hid_dim)
        if self.use_ocean_nodes:
            self.ocean_node_mlp = MLP(ocean_node_dim, hid_dim, hid_dim)
        self.mesh_node_mlp = MLP(mesh_node_dim, hid_dim, hid_dim)

        self.edge_mlp = MLP(edge_dim, hid_dim, hid_dim)

    def embed_edges(self, edge_attr):
        '''All edge features are currently the constant 1, so the MLP would return the same
        row for every edge. Compute that row once and broadcast it. The gate makes the
        shortcut disappear by itself once the edges carry real features.'''
        flat = edge_attr.reshape(-1, 1)
        if flat.numel() > 0 and bool((flat == 1).all()):
            return self.edge_mlp(flat[:1]).expand(flat.shape[0], -1)
        return self.edge_mlp(flat)

    def forward(self, data):

        data['grid'].x = self.grid_node_mlp(data['grid'].x)
        if self.use_ocean_nodes:
            data['ocean'].x = self.ocean_node_mlp(data['ocean'].x)
        data['mesh'].x = self.mesh_node_mlp(data['mesh'].x)
        data['grid', 'to', 'mesh'].edge_attr = self.embed_edges(data['grid', 'to', 'mesh'].edge_attr)
        if self.use_ocean_nodes:
            data['ocean', 'to', 'mesh'].edge_attr = self.embed_edges(data['ocean', 'to', 'mesh'].edge_attr)
        data['mesh', 'to', 'mesh'].edge_attr = self.embed_edges(data['mesh', 'to', 'mesh'].edge_attr)
        data['mesh', 'to', 'grid'].edge_attr = self.embed_edges(data['mesh', 'to', 'grid'].edge_attr)
        return data