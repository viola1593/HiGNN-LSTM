import torch
from torch import nn
import torch.nn.functional as F

from models.layers.embedder import Embedder
from models.layers.grid2mesh import Grid2MeshGNN
from models.layers.mesh2mesh import Mesh2MeshGNN
from models.layers.mesh2grid import Mesh2GridGNN     
from models.layers.output import Outputter   



class HiGNN_LSTM(nn.Module):
    def __init__(self, num_features, num_ocean_features, horizon,  
                 sequence_length=12, twsa_index=0, hidden_dim=128, num_processing_gnns=3,
                 gnn_embedding_dim=4,
                 rnn_hidden_size=32, 
                    rnn_layers=3, 
                    rnn_dropout=0.0,
                    seq_emb_length=4,
                    timestep_embed_dim=None, 
                    ):
        '''Hierarchical model for TWSA forecasting with GNN and RNN components.^
         Args:
             num_features (int): Number of input features.
             num_ocean_features (int): Number of ocean features.
             horizon (int): Number of timesteps to forecast aka forceasting horizon.
             sequence_length (int, optional): Length of the input sequence. Default is 12.
             twsa_index (int, optional): Index of TWSA in the input features. Default is 0.
             hidden_dim (int, optional): Hidden dimension for the GNN layers. Default is 128.
             num_processing_gnns (int, optional): Number of GNN layers in the processing step. Default is 3.
             gnn_embedding_dim (int, optional): Dimension of the output from GNN. Default is 4.
             rnn_layers (int, optional): Number of RNN layers. Default is 3.
             rnn_hidden_size (int, optional): Hidden size for the RNN layers. Default is 32.
             rnn_dropout (float, optional): Dropout for the RNN layers. Default is 0.0.
             seq_emb_length (int, optional): Length of the sequence embedding. Default is 4. Null means no temporal convolution of feature sequences.
             timestep_embed_dim (int, optional): Dimension of the timestep embedding. Default is None, which means no timestep embedding.

         '''
        super(HiGNN_LSTM, self).__init__()

        self.num_features = num_features
        self.num_ocean_features = num_ocean_features
        self.horizon = horizon         
        self.hidden_dim = hidden_dim
        self.num_processing_gnns = num_processing_gnns
        self.gnn_embedding_dim = gnn_embedding_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.rnn_dropout = rnn_dropout
        self.timestep_embed_dim = timestep_embed_dim
        self.seq_len = sequence_length
        self.seq_emb_size = seq_emb_length
        self.twsa_index = twsa_index

        # Define the layers/modules
        self.embedder = Embedder(grid_node_dim=self.seq_len*self.num_features, ocean_node_dim=self.seq_len*self.num_ocean_features,
                                  mesh_node_dim=3, edge_dim=1, hid_dim=self.hidden_dim)
        self.grid2mesh = Grid2MeshGNN(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.hidden_dim)
        
        if self.timestep_embed_dim is not None:
            self.timestep_embeddings = nn.Embedding(self.horizon, self.timestep_embed_dim)
        
            # FiLM adapters for timestep conditioning
            self.m2m_edge_film = nn.Linear(timestep_embed_dim, 2 * (self.hidden_dim + 2 * self.hidden_dim))  # scale + shift for input of edge MLP
            self.m2m_node_film = nn.Linear(timestep_embed_dim, 2 * (self.hidden_dim + self.hidden_dim))  # scale + shift for input of node MLP
        else: self.timestep_embeddings = None
        
        self.gnns = nn.ModuleList([Mesh2MeshGNN(self.hidden_dim, 
                                                self.hidden_dim, 
                                                self.hidden_dim, 
                                                timestep_dim=self.timestep_embed_dim) for _ in range(self.num_processing_gnns)])

        self.mesh2grid = Mesh2GridGNN(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.hidden_dim)
        self.outputter = Outputter(self.hidden_dim, self.hidden_dim, self.seq_len*self.gnn_embedding_dim)
        self.rnn = nn.LSTM(input_size=self.timestep_embed_dim+1+self.gnn_embedding_dim, hidden_size=self.rnn_hidden_size, 
                                num_layers=self.rnn_layers, batch_first=True, dropout=self.rnn_dropout)
        self.act = nn.ELU() 
        self.layer_norm = nn.LayerNorm(self.rnn_hidden_size)
        self.output_proj = nn.Linear(self.rnn_hidden_size, 1)


    def forward(self, data):

        data = data.clone() # absolutely need to clone it because forward changes of edge/node attrs
        batch_size = data['grid'].x.shape[0] # batch_size * num nodes
        twsa_in = data['grid'].x[ :, :, self.twsa_index].unsqueeze(-1) #  shape: [batch_size* num nodes, seq_len, 1]
        
        data['grid'].x = data['grid'].x.view(batch_size, -1) # collapse the time/feature dimension        
        data['ocean'].x = data['ocean'].x.view(data['ocean'].x.shape[0], -1) # collapse the time/feature dimension
        data = self.embedder(data)
        data = self.grid2mesh(data)

        # Store fixed TWSA history (don't update it during forecasting)
        twsa_history_fixed = twsa_in.clone()  # [batch_size*num_nodes, seq_len, 1]
        
        m2m_graphs = []
        twsa_forecast = []
        gnn_outputs = []

        
        
        for i in range(self.horizon):
            gnn_emb = data.clone() # cloning to get fresh data for each timestep
            timestep_embed = self.timestep_embeddings(torch.tensor(i, device=gnn_emb['grid'].x.device))# --> [batch_size* num_nodes, 1, hidden_dim/2]

            gnn_emb, gamma_edge, beta_edge, gamma_node, beta_node = self.shared_forward(gnn_emb, timestep_embed)
            
            gnn_emb = self.mesh2grid(gnn_emb, gamma_edge, beta_edge, gamma_node, beta_node) 
            gnn_output = self.outputter(gnn_emb) # shape gnn_output [batch_size*num_nodes, seq_len* gnn_embedding_dim]
            
            gnn_output = gnn_output.reshape(batch_size, self.seq_len, -1) # reshape to [batch_size*num_nodes, seq_len, gnn_embedding_dim]

            timestep_embed_expanded = timestep_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, self.seq_len, -1)
            input_ts = torch.cat((timestep_embed_expanded, twsa_history_fixed, gnn_output), dim=-1) # --> [batch_size*num_nodes, seq_len, 1+num_features+trigmonths(2)]
            h0 = torch.zeros(self.rnn_layers, batch_size, self.rnn_hidden_size).to(data['grid'].x.device)
            c0 = torch.zeros(self.rnn_layers, batch_size, self.rnn_hidden_size).to(data['grid'].x.device)
            hidden = (h0, c0)

            
            res, hidden = self.rnn(input_ts, hidden)
            res_norm = self.layer_norm(self.act(res[:, -1]))
            res_norm = self.output_proj(res_norm)

            # calculate twsa forecast from residual   
            twsa_out = res_norm + twsa_in[:, -1] # --> [batch_size* num_nodes, 1]

            twsa_forecast.append(twsa_out)            
            m2m_graphs.append(gnn_emb['mesh','to','mesh'].detach())
            gnn_outputs.append(gnn_output)


        # concatenate the forecast for all timesteps
        twsa_forecast = torch.cat(twsa_forecast, dim=-1) # --> [batch_size*num_nodes, horizon]
        return twsa_forecast, m2m_graphs, gnn_outputs  # return the gnn_outputs for every forecasting timestepization or further processing

   

   

    def shared_forward(self, gnn_emb, timestep_embed):
        '''use a timestep embedding with a single gnn for all timesteps'''
        if self.timestep_embeddings:
            gamma_edge, beta_edge = self.m2m_edge_film(timestep_embed).chunk(2, dim=-1)
            gamma_node, beta_node = self.m2m_node_film(timestep_embed).chunk(2, dim=-1)

        else:
            timestep_embed = None

        # Pass through the GNNs
        for gnn in self.gnns:
            gnn_emb = gnn(gnn_emb, gamma_edge, beta_edge, gamma_node, beta_node)
        return gnn_emb, gamma_edge, beta_edge, gamma_node, beta_node
            


    
    
    
