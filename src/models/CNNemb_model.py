'''CNN-based forecaster with TWSA embeddings and trig_months handling.


Not finished, but should be similar to the hierarchical model, but with a CNN instead of a GNN.'''

import torch
from torch import nn
from tsl.nn.layers.base.temporal_conv import TemporalConv



class CNN_forecaster(nn.Module):
    def __init__(self, num_features, horizon,  
                 sequence_length=12, twsa_index=0, hidden_dim=128, 
                 rnn_hidden_size=32, 
                    rnn_layers=3, 
                    rnn_dropout=0.0,
                    seq_emb_length=4,
                    timestep_embed_dim=None, 
                    method='simple'):
        super(CNN_forecaster, self).__init__()

        self.num_features = num_features-2 # remove trig_months
        self.horizon = horizon         
      
        self.num_processing_steps = self.horizon
        
        
        self.hidden_dim = hidden_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.rnn_dropout = rnn_dropout
        self.timestep_embed_dim = timestep_embed_dim
        self.method = method
        
        self.seq_len = sequence_length
        self.seq_emb_size = seq_emb_length
        self.twsa_index = twsa_index

        
        self.cnns = nn.ModuleList([nn.Conv2d(in_channels=self.num_features*self.seq_len, 
                                                out_channels=self.hidden_dim, 
                                                kernel_size=3, 
                                                padding='same') for _ in range(3)])

        self.rnn = torch.nn.LSTM(input_size=4, hidden_size=self.rnn_hidden_size, 
                                num_layers=self.rnn_layers, batch_first=True, dropout=self.rnn_dropout)
        self.output_proj = torch.nn.Linear(self.rnn_hidden_size, 1)
        # Introduce a residual gate to control the influence of the RNN output
        # Smart initialization that gradually reduces model confidence over time
        init_values = torch.linspace(2.0, -2.0, self.horizon)  # From ~0.88 to ~0.12 after sigmoid
        #self.residual_gates = nn.Parameter(init_values)

    def forward(self, data, teacher_forcing_ratio=0.0):
        data = data.clone()
        batch_size = data['grid'].x.shape[0] # batch_size * num nodes
        twsa_in = data['grid'].x[ :, :, self.twsa_index].unsqueeze(-1) #  shape: [batch_size* num nodes, seq_len, 1]
        trig_months = data['grid'].x[:, :, -2:] #  shape: [batch_size* num nodes, seq_len, 2]
        data['grid'].x = data['grid'].x[ :, :, :-2].reshape(batch_size, -1) # collapse the time/feature dimension        
        
        # Pass the data through the layers of the hierarchical GNN model
        # Encode temporal dimension of the input features
        # data = self.temp_conv(data) # --> [batch_size*num_nodes, seq_len, 1]

        data = self.embedder(data)
        data = self.grid2mesh(data)

        # Initialize the hidden state of the RNN
        h0 = torch.zeros(self.rnn_layers, batch_size, self.rnn_hidden_size).to(data['grid'].x.device)
        c0 = torch.zeros(self.rnn_layers, batch_size, self.rnn_hidden_size).to(data['grid'].x.device)
        hidden = (h0, c0)
        
        hetero_graphs = []
        twsa_forecast = []
        for i in range(self.horizon):
            gnn_emb = data.clone() # cloning to get fresh data for each timestep
            if self.method == 'simple':
                gnn_emb = self.shared_forward(gnn_emb, i)
            elif self.method == 'extra':
                gnn_emb = self.extra_forward(gnn_emb, i)
            
            gnn_emb = self.mesh2grid(gnn_emb) 
            gnn_output = self.outputter(gnn_emb) # shape gnn_output [batch_size*num_nodes, seq_len* 1]
            
            input_ts = torch.cat((twsa_in, trig_months, gnn_output.unsqueeze(-1)), dim=-1) # --> [batch_size*num_nodes, seq_len, 1+num_features+trigmonths(2)]
        
            res, hidden = self.rnn(input_ts, hidden)
            res = self.output_proj(res[:, -1])

            # calculate twsa forecast from residual   
            # alpha = torch.sigmoid(self.residual_gates[i]) 
            # twsa_out = (1 - alpha) * res +  alpha* twsa_in[:, -1]
            twsa_out = res + twsa_in[:, -1] # --> [batch_size* num_nodes, 1]

            twsa_forecast.append(twsa_out)            
            hetero_graphs.append(gnn_emb)

            #calculate trig_month for next autoregressive step
            next_trig_month = self.next_trig_month(trig_months[:, -1,0], trig_months[:, -1,1]) # --> [batch_size*nodes, 1,  2]
            trig_months = torch.cat((trig_months[:, 1:], next_trig_month.reshape(-1,1,2)), dim=1)

            # append input twsa_ts for next autoregressive step    
            # Decide whether to use ground truth or prediction (teacher forcing)
            if teacher_forcing_ratio > 0:
                # Use teacher forcing with the specified probability
                use_teacher_forcing = (torch.rand(1).item() < teacher_forcing_ratio)
                next_step = data['grid'].y[:, i] if use_teacher_forcing else twsa_out
            else:
                # Use the predicted value
                next_step = twsa_out
            twsa_in = torch.cat((twsa_in[:, 1:], next_step.reshape(-1,1,1)), dim=1)
        
        # concatenate the forecast for all timesteps
        twsa_forecast = torch.cat(twsa_forecast, dim=-1) # --> [batch_size*num_nodes, horizon]
        return twsa_forecast, hetero_graphs

    def extra_forward(self, gnn_emb, i):
        '''Use n gnns separte gnns for each timestep'''
        gnn_emb = self.gnns[2*i](gnn_emb)
        gnn_emb = self.gnns[2*i+1](gnn_emb)      
        return gnn_emb

    def shared_forward(self, gnn_emb, i):
        '''use a timestep embedding with a single gnn for all timesteps'''
        # Add the timestep embedding to the data
        timestep_embed = self.timestep_embeddings(torch.tensor(i, device=gnn_emb['grid'].x.device))# --> [batch_size* num_nodes, 1, hidden_dim/2]
        
        for gnn in self.gnns:
            gnn_emb = gnn(gnn_emb, timestep_embed)
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
    
    
