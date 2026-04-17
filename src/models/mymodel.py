import math
import torch
from torch_geometric.nn import GCNConv
# from torch_geometric.nn.conv import GATConv
from tsl.nn.layers.graph_convs.gat_conv import GATConv
from tsl.nn.blocks.encoders.recurrent.rnn import RNN
from tsl.nn.layers.base.temporal_conv import TemporalConv



""" this model should be able to forecast TWSA autoregressively. 
for each number of delay months between the input time series we use a separate GNN model to encode spatio temporal connections between the nodes aka grid cells.
"""


class TWSA_forecaster(torch.nn.Module):
    def __init__(self, num_features, horizon, num_nodes=22523, num_months_per_gnn=None, 
                 sequence_length=12, twsa_index=0, rnn_hidden_size=32):
        super(TWSA_forecaster, self).__init__()
        '''TWSA index default. 0'''
        
        self.num_features = num_features-2 # remove trig_months
        self.horizon = horizon 
        self.num_nodes = num_nodes
        self.num_months_per_gnn = num_months_per_gnn
        if num_months_per_gnn is not None:
            self.num_gnns = math.ceil(self.horizon/self.num_months_per_gnn)
            self.gnn_indices = [i for i in range(self.num_gnns) for _ in range(self.num_months_per_gnn)]
            self.gnn_indices = self.gnn_indices[:self.horizon]
        else:
            self.num_gnns = self.horizon
            self.gnn_indices = [i for i in range(self.horizon)]
        

        self.rnn_hidden_size = rnn_hidden_size
        
        self.seq_len = sequence_length
        self.seq_emb_size = 4
        self.twsa_index = twsa_index
        #self.read_out_emb = 12


        self.tp_embedding = TemporalConv(input_channels=self.seq_len, output_channels=self.seq_emb_size, kernel_size=3, channel_last=False)
        # self.grid2mesh = Grid2Mesh(hidden_dim, hidden_dim)
        self.gnns = torch.nn.ModuleList([GCNConv(self.num_features*self.seq_emb_size, self.num_features*self.seq_emb_size) for _ in range(self.num_gnns)])
        self.act = torch.nn.ELU()
        # self.mesh2grid = Mesh2Grid(hidden_dim, hidden_dim)
        self.gnn_readout = torch.nn.ModuleList([torch.nn.Linear(self.num_features*self.seq_emb_size, self.seq_len) for _ in range(self.num_gnns)])
        self.rnn = torch.nn.LSTM(input_size=4, hidden_size=self.rnn_hidden_size, 
                                num_layers=3, batch_first=True)
          # Add output projection
        self.output_proj = torch.nn.Linear(self.rnn_hidden_size, 1)
        #self.dropout = torch.nn.Dropout(p=0.5)
        self.batch_norm = torch.nn.BatchNorm1d(self.num_nodes) # TODO: is number of nodes correct?

    def forward(self, x, edge_index):
        
        '''input shape: [batch_size, num_timesteps, num_nodes, num_features], '''
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        num_nodes = x.shape[2]
        num_features = x.shape[3]

        twsa_ts = x[:, :, :, self.twsa_index].unsqueeze(-1) #  shape: [batch_size, num_timesteps, num_nodes, 1]
        trig_months = x[:, :, :, -2:] #  shape: [batch_size, num_timesteps, num_nodes, 2]
        x = x[:, :, :, :-2] #  shape: [batch_size, num_timesteps, num_nodes, num_features-2]
        twsa_forecast = []
        
        x = self.tp_embedding(x) # --> [batch_size, 4, num_nodes, num_features]
        #print(x.shape)
        seq_len = x.shape[1]
        x = x.permute(0,2,1,3).reshape(batch_size, num_nodes, -1) # --> [batch_size, num_nodes, num_emb_timesteps*num_features]
        hn = None
   
        
        for i in self.gnn_indices:
            #x = self.batch_norm(x)
            
            gnn_emb = self.gnns[i](x, edge_index) # for every time step we get another graph representation
            gnn_emb = self.act(gnn_emb)
            #
            # x = self.dropout(x)
            gnn_emb = self.gnn_readout[i](gnn_emb).permute(0,2,1).unsqueeze(-1) # --> [batch_size, self.seq_len,  num_nodes, 1]
            
            
            #import pdb; pdb.set_trace()
            input_ts = torch.cat((twsa_ts, trig_months, gnn_emb), dim=-1).permute(0,2,1,3).reshape(batch_size*num_nodes, self.seq_len, -1) # --> [batch_size*num_nodes, num_timesteps, 1+num_output_feat from gnn]
            res, hn = self.rnn(input_ts, hn)
            
            res = self.output_proj(res)
            
            res = res[:, -1, :].reshape(batch_size, num_nodes,  1)

            next_step = res+ twsa_ts[:, -1] # --> [batch_size, num_nodes, 1]
            twsa_forecast.append(next_step)
           
            next_time = self.next_trig_month(trig_months[:, -1,:, 0], trig_months[:, -1,:,1]) # --> [batch_size, 1, nodes, 2]
            trig_months = torch.cat((trig_months[:, 1:], next_time.unsqueeze(1)), dim=1)
            twsa_ts = torch.cat((twsa_ts[:, 1:], next_step.permute(0,2,1).unsqueeze(-1)), dim=1)

            # twsa_ts = torch.cat((twsa_ts[:, 1:], next_step.permute(0,2,1)), dim=1)
        
        twsa_forecast = torch.cat(twsa_forecast, dim=-1).permute(0,2,1) # --> [batch_size, horizon, num_nodes]

        return twsa_forecast
    

    def next_trig_month(self, last_cos_time, last_sine_time):
        period = 12
           # Get angle in radians using arctan2 (handles quadrants correctly)
        angle = torch.atan2(last_sine_time, last_cos_time)
        
        # Convert angle to month (1-12)
        # Shift from [-π,π] to [0,2π] range if needed
        angle = torch.where(angle < 0, angle + 2*torch.pi, angle)
        
        # Convert to months (angle * 12/(2π))
        next_month = (torch.round(angle * 12 / (2*torch.pi)) +1) % 12
        
        new_cos_time = torch.cos(2*torch.pi*(next_month)/period).unsqueeze(-1) # --> [batch_size, 1, nodes, 1] 
        new_sine_time = torch.sin(2*torch.pi*(next_month)/period).unsqueeze(-1) # --> [batch_size, 1, nodes, 1]

        
        return torch.cat((new_cos_time, new_sine_time), dim=-1)