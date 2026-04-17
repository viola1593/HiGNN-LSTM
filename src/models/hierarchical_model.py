import torch
from torch import nn
import torch.nn.functional as F
from tsl.nn.layers.base.temporal_conv import TemporalConv
# from torch_geometric.nn.models.basic_gnn import GAT
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from models.layers.gat import Mesh2MeshGAT

from models.layers.tempconv import TemporalEncoder
from models.layers.embedder import Embedder
from models.layers.grid2mesh import Grid2MeshGNN
from models.layers.mesh2mesh import Mesh2MeshGNN
from models.layers.mesh2grid import Mesh2GridGNN     
from models.layers.output import Outputter   
from models.layers.mlp import MLP


class HM_forecaster_auto(nn.Module):
    def __init__(self, num_features, num_ocean_features, horizon,  
                 sequence_length=12, twsa_index=0, hidden_dim=128, num_processing_gnns=3,
                 gnn_embedding_dim=4,
                 rnn_hidden_size=32, 
                    rnn_layers=3, 
                    rnn_dropout=0.0,
                    seq_emb_length=4,
                    timestep_embed_dim=None, 
                    method='simple',
                    num_heads=8):
        super(HM_forecaster_auto, self).__init__()

        self.num_features = num_features#-2 # remove trig_months
        self.num_ocean_features = num_ocean_features
        self.horizon = horizon         
      
        self.num_processing_steps = self.horizon
        
        
        self.hidden_dim = hidden_dim
        self.num_processing_gnns = num_processing_gnns
        self.gnn_embedding_dim = gnn_embedding_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.rnn_dropout = rnn_dropout
        self.timestep_embed_dim = timestep_embed_dim
        self.method = method
        self.attention_weights=None
        
        self.seq_len = sequence_length
        self.seq_emb_size = seq_emb_length
        self.twsa_index = twsa_index


        self.embedder = Embedder(grid_node_dim=self.seq_len*self.num_features, ocean_node_dim=self.seq_len*self.num_ocean_features,
                                  mesh_node_dim=3, edge_dim=1, hid_dim=self.hidden_dim)

        self.grid2mesh = Grid2MeshGNN(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.hidden_dim)
        if self.method =='simple':
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
        # self.outputter = Outputter(self.hidden_dim, self.hidden_dim/2, self.seq_len*self.gnn_embedding_dim)
        
        # self.out_mlp = MLP(in_dim=self.seq_len*self.gnn_embedding_dim, hid_dim=32, out_dim=1)
        
        self.rnn = nn.LSTM(input_size=self.timestep_embed_dim+1+self.gnn_embedding_dim, hidden_size=self.rnn_hidden_size, 
                                num_layers=self.rnn_layers, batch_first=True, dropout=self.rnn_dropout)
        self.act = nn.ELU() # or SELU
        self.layer_norm = nn.LayerNorm(self.rnn_hidden_size)
        self.output_proj = nn.Linear(self.rnn_hidden_size, 1) # project to horizon size (seq2seq) or 1 if autoregressive


    def forward(self, data, teacher_forcing_ratio=0.0):

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
        # attentions_per_timestep = [] # attention weights for visualization
        
        
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

            # next_step = twsa_out
            # twsa_in = torch.cat((twsa_in[:, 1:], next_step.reshape(-1,1,1)), dim=1)

        
        # concatenate the forecast for all timesteps
        twsa_forecast = torch.cat(twsa_forecast, dim=-1) # --> [batch_size*num_nodes, horizon]
        # self.attention_weights = attentions_per_timestep
        return twsa_forecast, m2m_graphs, gnn_outputs  # return the gnn_outputs for every forecasting timestepization or further processing

   

    def extra_forward(self, gnn_emb, i):
        '''Use n gnns separte gnns for each timestep'''
        gnn_emb = self.gnns[2*i](gnn_emb)
        gnn_emb = self.gnns[2*i+1](gnn_emb)      
        return gnn_emb, None  # no attention weights in this case, return None
    
    def gat_forward(self, gnn_emb, i, attentions: list):
        '''Use n gnns separte gnns for each timestep'''
        if self.timestep_embeddings:
            timestep_embed = self.timestep_embeddings(torch.tensor(i, device=gnn_emb['grid'].x.device))# --> [batch_size* num_nodes, 1, hidden_dim/2]
            gamma_node, beta_node = self.m2m_node_film(timestep_embed).chunk(2, dim=-1)
        else:
            timestep_embed = None
        
        for gnn in self.gnns:
            gnn_emb, (edges, att) = gnn(gnn_emb, gamma_node, beta_node)
        attentions.append(att.detach())
        return gnn_emb, attentions

    def shared_forward(self, gnn_emb, timestep_embed):
        '''use a timestep embedding with a single gnn for all timesteps'''
        if self.timestep_embeddings:
            gamma_edge, beta_edge = self.m2m_edge_film(timestep_embed).chunk(2, dim=-1)
            gamma_node, beta_node = self.m2m_node_film(timestep_embed).chunk(2, dim=-1)

        else:
            timestep_embed = None
        # timestep_embed = F.one_hot(torch.tensor(i, device=gnn_emb['grid'].x.device), num_classes=self.horizon).bfloat16()
        # timestep_embed = torch.Tensor([i/self.horizon]).to(gnn_emb['grid'].x.device)
        # Pass through the GNNs
        for gnn in self.gnns:
            gnn_emb = gnn(gnn_emb, gamma_edge, beta_edge, gamma_node, beta_node)
        return gnn_emb, gamma_edge, beta_edge, gamma_node, beta_node
            


    def next_trig_month(self, last_cos_time, last_sine_time):
        period = 12
        # Get angle in radians using arctan2 (handles quadrants correctly)
        angle = torch.atan2(last_sine_time, last_cos_time)
        
        # Convert angle to month (1-12)
        # Shift from [-π,π] to [0,2π] range if needed
        angle = torch.where(angle < 0, angle + 2*torch.pi, angle)
        
        # Convert to months (angle * 12/(2π))
        next_month = (torch.round(angle * 12 / (2*torch.pi)) +1) % 12
        
        new_cos_time = torch.cos(2*torch.pi*(next_month)/period).unsqueeze(-1) # --> [batch_size* nodes, 1, 1] 
        new_sine_time = torch.sin(2*torch.pi*(next_month)/period).unsqueeze(-1) # --> [batch_size* nodes, 1, 1]

        
        return torch.cat((new_cos_time, new_sine_time), dim=-1)
    
    
class HM_forecaster_seq2seq(nn.Module):
    def __init__(self, num_features, horizon,  
                 sequence_length=12, twsa_index=0, hidden_dim=128, num_processing_gnns=3,
                 gnn_embedding_dim=4,
                 rnn_hidden_size=32, 
                    rnn_layers=3, 
                    rnn_dropout=0.0,
                    seq_emb_length=4,
                    timestep_embed_dim=None, 
                    method='simple',
                    num_heads=8):
        super(HM_forecaster_seq2seq, self).__init__()

        self.num_features = num_features#-2 # remove trig_months
        self.horizon = horizon         
      
        self.num_processing_steps = self.horizon
        
        
        self.hidden_dim = hidden_dim
        self.num_processing_gnns = num_processing_gnns
        self.gnn_embedding_dim = gnn_embedding_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.rnn_dropout = rnn_dropout
        self.timestep_embed_dim = timestep_embed_dim
        self.method = method
        
        self.seq_len = sequence_length
        self.seq_emb_size = seq_emb_length
        self.twsa_index = twsa_index
        self.attention = None

        if self.seq_emb_size is not None:
            self.temp_conv = TemporalEncoder(in_channels=self.num_features,  in_seq_len=self.seq_len, out_seq_len=self.seq_emb_size)

            # embedder casts all the input data to the hidden dimension
            self.embedder = Embedder(grid_node_dim=self.seq_emb_size*self.num_features, mesh_node_dim=3, edge_dim=1, hid_dim=self.hidden_dim)
        else:
            self.embedder = Embedder(grid_node_dim=self.seq_len*self.num_features, mesh_node_dim=3, edge_dim=1, hid_dim=self.hidden_dim)

        self.grid2mesh = Grid2MeshGNN(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.hidden_dim)
        
        
        if self.method =='simple':
            self.gnns = torch.nn.ModuleList([Mesh2MeshGNN(self.hidden_dim, 
                                                          self.hidden_dim, 
                                                          self.hidden_dim, 
                                                          timestep_dim=None) for _ in range(self.num_processing_gnns)])


        elif self.method == 'gat':
            from torch_geometric.nn.models.basic_gnn import GAT
            self.gnns = nn.ModuleList([
                GAT(in_channels=self.hidden_dim, hidden_channels=self.hidden_dim, num_layers=self.num_processing_gnns,
                    v2=True, heads=num_heads) 
                for _ in range(self.num_processing_gnns)
            ])
        else: print(f'Unknown method {self.method}')

        self.mesh2grid = Mesh2GridGNN(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.hidden_dim)

        # self.outputter = Outputter(self.hidden_dim, self.hidden_dim, self.seq_len*self.gnn_embedding_dim)
        self.outputter = Outputter(self.hidden_dim, self.hidden_dim, self.horizon)

        self.rnn = torch.nn.LSTM(input_size=3+self.gnn_embedding_dim, hidden_size=self.rnn_hidden_size, 
                                num_layers=self.rnn_layers, batch_first=True, dropout=self.rnn_dropout)
        self.act = torch.nn.ELU()
        self.layer_norm = torch.nn.LayerNorm(self.rnn_hidden_size)
        self.output_proj = torch.nn.Linear(self.rnn_hidden_size, self.horizon) # project to horizon size (seq2seq) or 1 if autoregressive


    
    def forward(self, data, teacher_forcing_ratio=None):
        
        data = data.clone() # absolutely need to clone it because forward changes of edge/node attrs
        batch_size = data['grid'].x.shape[0] # batch_size * num nodes
        twsa_in = data['grid'].x[ :, :, self.twsa_index].unsqueeze(-1) #  shape: [batch_size* num nodes, seq_len, 1]
        trig_months = data['grid'].x[:, :, -2:] #  shape: [batch_size* num nodes, seq_len, 2]
        # data['grid'].x = data['grid'].x.reshape(batch_size, -1) # collapse the time/feature dimension        
        # data['grid'].x = data['grid'].x[ :, :, :-2].reshape(batch_size, -1) # collapse the time/feature dimension and remove trig_months        

        data = self.temp_conv(data)
        data['grid'].x = data['grid'].x.reshape(batch_size, -1) # collapse the time/feature dimension    
        
        data = self.embedder(data)
        data = self.grid2mesh(data)

        # Initialize the hidden state of the RNN
        h0 = torch.zeros(self.rnn_layers, batch_size, self.rnn_hidden_size).to(data['grid'].x.device)
        c0 = torch.zeros(self.rnn_layers, batch_size, self.rnn_hidden_size).to(data['grid'].x.device)
        
        
        gnn_emb = data.clone() # cloning to get fresh data for each timestep
        if self.method == 'simple':
            gnn_emb = self.mlp_forward(gnn_emb)

        elif self.method == 'gat':
            gnn_emb = self.gat_forward(gnn_emb)
        
        gnn_emb = self.mesh2grid(gnn_emb) 
        res_norm = self.outputter(gnn_emb) # shape gnn_output [batch_size*num_nodes, seq_len*1]
        # gnn_output = gnn_output.reshape(batch_size, self.seq_len, -1) # reshape to [batch_size*num_nodes, seq_len, gnn_emb_dim]
        # input_ts = torch.cat((twsa_in, trig_months, gnn_output), dim=-1) # --> [batch_size*num_nodes, seq_len, 1+num_features+trigmonths(2)]
        # # input_ts = torch.cat((twsa_in, trig_months), dim=-1) # --> [batch_size*num_nodes, seq_len, 1+num_features+trigmonths(2)]
      

        # res, hidden = self.rnn(input_ts, (h0, c0))
        # res_norm = self.layer_norm(self.act(res[:, -1]))
        # res_norm = self.output_proj(res_norm) # --> [batch_size*num_nodes, horizon]
        # calculate twsa forecast from residual
        twsa_forecast = res_norm + twsa_in[:, -1] # --> [batch_size* num_nodes, horizon]

        
        hetero_graphs = [gnn_emb['mesh','to','mesh'].detach().cpu() for _ in range(self.horizon)]
        return twsa_forecast, hetero_graphs, None



    
    def gat_forward(self, gnn_emb):
        '''pass through the GAT'''

        for gnn in self.gnns:
                x  = gnn(gnn_emb['mesh'].x, 
                                gnn_emb['mesh', 'to', 'mesh'].edge_index, 
                                gnn_emb['mesh', 'to', 'mesh'].edge_attr)
                gnn_emb['mesh'].x = gnn_emb['mesh'].x + x
        return gnn_emb

    def mlp_forward(self, gnn_emb):
        '''pass through the mesh2mesh gnn'''
        for gnn in self.gnns:
            gnn_emb = gnn(gnn_emb)
        return gnn_emb
            


    def next_trig_month(self, last_cos_time, last_sine_time):
        period = 12
        # Get angle in radians using arctan2 (handles quadrants correctly)
        angle = torch.atan2(last_sine_time, last_cos_time)
        
        # Convert angle to month (1-12)
        # Shift from [-π,π] to [0,2π] range if needed
        angle = torch.where(angle < 0, angle + 2*torch.pi, angle)
        
        # Convert to months (angle * 12/(2π))
        next_month = (torch.round(angle * 12 / (2*torch.pi)) +1) % 12
        
        new_cos_time = torch.cos(2*torch.pi*(next_month)/period).unsqueeze(-1) # --> [batch_size* nodes, 1, 1] 
        new_sine_time = torch.sin(2*torch.pi*(next_month)/period).unsqueeze(-1) # --> [batch_size* nodes, 1, 1]

        
        return torch.cat((new_cos_time, new_sine_time), dim=-1)