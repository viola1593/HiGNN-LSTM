import copy
import torch
from torch import nn
import torch.nn.functional as F

from models.layers.embedder import Embedder
from models.layers.grid2mesh import Grid2MeshGNN
from models.layers.mesh2mesh import Mesh2MeshGNN
from models.layers.mesh2grid import Mesh2GridGNN     
from models.layers.output import Outputter   



class HiGNN_LSTM(nn.Module):
    def __init__(self, num_features, num_ocean_features, horizon, timestep_embed_dim,
                 sequence_length=12, twsa_index=0, hidden_dim=128, num_processing_gnns=3,
                 gnn_embedding_dim=4,
                 skip_gnn=False,
                 rnn_hidden_size=32,
                    rnn_layers=3,
                    rnn_dropout=0.0,
                    use_ocean_nodes=True,
                    aggregation='sum',
                    residual_from_last=True,
                    ):
        '''Hierarchical model for TWSA forecasting with GNN and RNN components.^
         Args:
             num_features (int): Number of input features.
             num_ocean_features (int): Number of ocean features.
             horizon (int): Number of timesteps to forecast aka forceasting horizon.
             timestep_embed_dim (int): Dimension of the timestep embedding. Required, the model
                is always conditioned on the forecasting timestep.
             sequence_length (int, optional): Length of the input sequence. Default is 12.
             twsa_index (int, optional): Index of TWSA in the input features. Default is 0.
             hidden_dim (int, optional): Hidden dimension for the GNN layers. Default is 128.
             num_processing_gnns (int, optional): Number of GNN layers in the processing step. Default is 3.
             gnn_embedding_dim (int, optional): Dimension of the output from GNN. Default is 4.
             skip_gnn (bool, optional): Whether to use skip the GNNs for each timestep. Used for ablation studies.Default is False.
             rnn_layers (int, optional): Number of RNN layers. Default is 3.
             rnn_hidden_size (int, optional): Hidden size for the RNN layers. Default is 32.
             rnn_dropout (float, optional): Dropout for the RNN layers. Default is 0.0.
             use_ocean_nodes (bool, optional): Whether the graph contains ocean nodes. If False, no ocean
                embedding/message passing is built at all. Used for the no_ocean graph variant. Default is True.
             aggregation (str, optional): 'sum' or 'mean', how incoming messages are reduced in the
                grid2mesh, mesh2mesh and mesh2grid layers. Default is 'sum'.
             residual_from_last (bool, optional): Whether the forecast is a residual on top of the last input
                TWSA. If False, TWSA is predicted directly (e.g. anomalies, zero output = climatology). Default is True.

         '''
        super(HiGNN_LSTM, self).__init__()

        if timestep_embed_dim is None:
            raise ValueError('timestep_embed_dim is required, the model is always conditioned '
                             'on the forecasting timestep')

        self.num_features = num_features
        self.num_ocean_features = num_ocean_features
        self.horizon = horizon         
        self.hidden_dim = hidden_dim
        self.num_processing_gnns = num_processing_gnns
        self.skip_gnn = skip_gnn
        self.gnn_embedding_dim = gnn_embedding_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.rnn_dropout = rnn_dropout
        self.timestep_embed_dim = timestep_embed_dim
        self.seq_len = sequence_length
        self.twsa_index = twsa_index
        self.use_ocean_nodes = use_ocean_nodes
        self.residual_from_last = residual_from_last

        # Define the layers/modules
        self.embedder = Embedder(grid_node_dim=self.seq_len*self.num_features, ocean_node_dim=self.seq_len*self.num_ocean_features,
                                  mesh_node_dim=3, edge_dim=1, hid_dim=self.hidden_dim,
                                  use_ocean_nodes=self.use_ocean_nodes)
        self.grid2mesh = Grid2MeshGNN(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.hidden_dim,
                                      use_ocean_nodes=self.use_ocean_nodes, aggregation=aggregation)

        self.timestep_embeddings = nn.Embedding(self.horizon, self.timestep_embed_dim)

        # FiLM adapters for timestep conditioning
        self.m2m_edge_film = nn.Linear(timestep_embed_dim, 2 * (self.hidden_dim + 2 * self.hidden_dim))  # scale + shift for input of edge MLP
        self.m2m_node_film = nn.Linear(timestep_embed_dim, 2 * (self.hidden_dim + self.hidden_dim))  # scale + shift for input of node MLP
        # zero-init so that at init gamma=beta=0, i.e. (1+gamma)*x+beta is the identity and
        # the timestep conditioning is learned as a perturbation of the unconditioned model
        for film in (self.m2m_edge_film, self.m2m_node_film):
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)

        self.gnns = nn.ModuleList([Mesh2MeshGNN(self.hidden_dim, 
                                                self.hidden_dim, 
                                                self.hidden_dim,
                                                aggregation=aggregation) for _ in range(self.num_processing_gnns)])

        self.mesh2grid = Mesh2GridGNN(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.hidden_dim, aggregation=aggregation)
        self.outputter = Outputter(self.hidden_dim, self.hidden_dim, self.seq_len*self.gnn_embedding_dim)
        self.rnn = nn.LSTM(input_size=self.timestep_embed_dim+1+self.gnn_embedding_dim, hidden_size=self.rnn_hidden_size, 
                                num_layers=self.rnn_layers, batch_first=True, dropout=self.rnn_dropout)
        self.act = nn.ELU() 
        self.layer_norm = nn.LayerNorm(self.rnn_hidden_size)
        self.output_proj = nn.Linear(self.rnn_hidden_size, 1)


    def forward(self, data):

        data = copy.copy(data) # need own attribute dicts because forward replaces edge/node attrs,
                               # but the replaced tensors are new ones, so no deep copy is needed
        batch_size = data['grid'].x.shape[0] # batch_size * num nodes
        twsa_in = data['grid'].x[ :, :, self.twsa_index].unsqueeze(-1) #  shape: [batch_size* num nodes, seq_len, 1]
        
        data['grid'].x = data['grid'].x.view(batch_size, -1) # collapse the time/feature dimension        
        if self.use_ocean_nodes:
            data['ocean'].x = data['ocean'].x.view(data['ocean'].x.shape[0], -1) # collapse the time/feature dimension
        data = self.embedder(data)
        data = self.grid2mesh(data)

        # Store fixed TWSA history (don't update it during forecasting)
        twsa_history_fixed = twsa_in.clone()  # [batch_size*num_nodes, seq_len, 1]
        
        m2m_graphs = []
        twsa_forecast = []
        gnn_outputs = []

        
        
        for i in range(self.horizon):
            gnn_emb = copy.copy(data) # fresh attribute dicts, so each timestep starts from the
                                      # post-grid2mesh state; the GNNs only rebind x/edge_attr,
                                      # so the shared tensors are never written into
            timestep_embed = self.timestep_embeddings(torch.tensor(i, device=gnn_emb['grid'].x.device))# --> [batch_size* num_nodes, 1, hidden_dim/2]
            if not self.skip_gnn:
                gnn_emb, gamma_edge, beta_edge, gamma_node, beta_node = self.shared_forward(gnn_emb, timestep_embed)    
                gnn_emb = self.mesh2grid(gnn_emb, gamma_edge, beta_edge, gamma_node, beta_node) 
            gnn_output = self.outputter(gnn_emb) # shape gnn_output [batch_size*num_nodes, seq_len* gnn_embedding_dim]
            
            gnn_output = gnn_output.reshape(batch_size, self.seq_len, -1) # reshape to [batch_size*num_nodes, seq_len, gnn_embedding_dim]

            timestep_embed_expanded = timestep_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, self.seq_len, -1)
            input_ts = torch.cat((timestep_embed_expanded, twsa_history_fixed, gnn_output), dim=-1) # --> [batch_size*num_nodes, seq_len, 1+num_features+trigmonths(2)]
            # hx=None makes nn.LSTM start from zeros, which is what h0/c0 were; the returned
            # hidden state is never used because every step starts from zeros again.
            res, _ = self.rnn(input_ts, None)
            res_norm = self.layer_norm(self.act(res[:, -1]))
            res_norm = self.output_proj(res_norm)

            # calculate twsa forecast from residual (or directly without the last-input base)
            twsa_out = res_norm + twsa_in[:, -1] if self.residual_from_last else res_norm # --> [batch_size* num_nodes, 1]

            twsa_forecast.append(twsa_out)            
            m2m_graphs.append(gnn_emb['mesh','to','mesh'].detach())
            gnn_outputs.append(gnn_output)


        # concatenate the forecast for all timesteps
        twsa_forecast = torch.cat(twsa_forecast, dim=-1) # --> [batch_size*num_nodes, horizon]
        return twsa_forecast, m2m_graphs, gnn_outputs  # return the gnn_outputs for every forecasting timestepization or further processing

   

   

    def shared_forward(self, gnn_emb, timestep_embed):
        '''use a timestep embedding with a single gnn for all timesteps'''
        gamma_edge, beta_edge = self.m2m_edge_film(timestep_embed).chunk(2, dim=-1)
        gamma_node, beta_node = self.m2m_node_film(timestep_embed).chunk(2, dim=-1)

        # Pass through the GNNs
        for gnn in self.gnns:
            gnn_emb = gnn(gnn_emb, gamma_edge, beta_edge, gamma_node, beta_node)
        return gnn_emb, gamma_edge, beta_edge, gamma_node, beta_node
            


    
    
    
