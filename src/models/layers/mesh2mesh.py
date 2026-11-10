import torch
from torch import nn
import torch_geometric as pyg
from .mlp import MLP

'''Adapted from: https://github.com/corentin-dfg/graph4sen2dwater/blob/main/models/graphcast/mesh2mesh.py
FiLM modulation and EdgePruner from Copliot (GPT-4.1)'''


class Mesh2MeshGNN(nn.Module):
    def __init__(self, mesh_node_dim, edge_dim, hid_dim, aggregation='sum'):
        super().__init__()
        self.aggregation = aggregation
        self.edge_mlp = MLP(edge_dim+mesh_node_dim+mesh_node_dim, hid_dim, hid_dim)
        self.mesh_mlp = MLP(mesh_node_dim+hid_dim, hid_dim, hid_dim)


    def forward(self, data, gamma_edge=None, beta_edge=None, gamma_node=None, beta_node=None):
        edge_feats = data['mesh', 'to', 'mesh'].edge_attr
        source_nodes = data['mesh', 'to', 'mesh'].edge_index[0]
        target_nodes = data['mesh', 'to', 'mesh'].edge_index[1]

        # Get original features
        src_feats = data['mesh'].x[source_nodes]
        tgt_feats = data['mesh'].x[target_nodes]

        # Concatenate features
        edge_input = torch.cat([edge_feats, src_feats, tgt_feats], dim=-1)
        
        # FiLM modulation for edge MLP input
        if gamma_edge is not None:
            edge_input = (1 + gamma_edge) * edge_input + beta_edge

        
        edge_new = self.edge_mlp(edge_input)

        # Node update
        agg_edge = pyg.utils.scatter(edge_new, target_nodes, dim=0, reduce=self.aggregation, dim_size=data['mesh'].x.shape[0])
        node_input = torch.cat([data['mesh'].x, agg_edge], dim=-1)

        # FiLM modulation for edge MLP input
        if gamma_node is not None:
            node_input = (1 + gamma_node) * node_input + beta_node

        mesh_new = self.mesh_mlp(node_input)

        data['mesh'].x =  data['mesh'].x + mesh_new #
        data['mesh', 'to', 'mesh'].edge_attr = data['mesh', 'to', 'mesh'].edge_attr + edge_new
        return data


