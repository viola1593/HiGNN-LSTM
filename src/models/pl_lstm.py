import torch
import torch.nn as nn
# from torch.nn import LSTM
import pytorch_lightning as pl
from torch.nn import MSELoss

class LSTMRegressor(pl.LightningModule):
    '''
    from https://www.kaggle.com/code/tartakovsky/pytorch-lightning-lstm-timeseries-clean-code
    '''
    def __init__(self, 
                 n_features: int=None, 
                 hidden_size: int=None, 
                 seq_len: int=None, 
                 batch_size: int=None,
                 num_layers: int=None, 
                 dropout: float=0.0, 
                 learning_rate: float=None,
                 criterion=MSELoss()):
        super(LSTMRegressor, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate

        self.lstm = nn.LSTM(input_size=n_features, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:,-1])
        return y_pred
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch['x_local'], batch['y_local']
        #x = x.permute(0, 2, 1)
        y_hat = self(x)
        y_hat = y_hat.squeeze()
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        
        x, y = batch['x_local'], batch['y_local']
        #x = x.permute(0, 2, 1)
        y_hat = self(x)
        y_hat = y_hat.squeeze()
        loss = self.criterion(y_hat, y)
        
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch['x_local'], batch['y_local']
        #x = x.permute(0, 2, 1)
        y_hat = self(x)
        y_hat = y_hat.squeeze()
        loss = self.criterion(y_hat, y)
        
        self.log('test_loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y, lonlat = batch['x_local'], batch['y_local'], batch['pos_local']
        #x = x.permute(0, 2, 1)
        y_hat = self(x)
        y_hat = y_hat.squeeze()
        return y_hat, y, x # returning inputs x as well for plotting purposes