import warnings

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.regression import MeanSquaredError
from torchmetrics import ExplainedVariance



from models.layers.convlstm_model import ConvLSTM  


class ConvLSTMModule(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, horizon, kernel_size, num_layers, learning_rate=1e-3, lr_scheduler=None,
                 residual_from_last=True):
        super(ConvLSTMModule, self).__init__()

        self.horizon = horizon
        self.residual_from_last = residual_from_last # False: residuals accumulate from 0 instead of the last input twsa
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler # None | 'cosine'

        self.convlstm = ConvLSTM(input_dim=input_dim, 
                                 hidden_dim=hidden_dim, 
                                 kernel_size=kernel_size, 
                                 num_layers=num_layers, 
                                 batch_first=True, 
                                 bias=True, 
                                 return_all_layers=False)
        self.output_layer = nn.Linear(hidden_dim, self.horizon)
        

        self.criterion = nn.MSELoss()
        self.metrics = nn.ModuleDict({'mse' : nn.MSELoss(),
                        'rmse': MeanSquaredError(squared=False), # RMSE
                        'explained_var': ExplainedVariance(),
                        })
        
    def forward(self, x):
        # x should be of shape (batch_size, seq_len, channels, height, width)
        # last_state_list is a list of tuples (h, c) for hidden state and memory, 
        # h and c are of shape (batch_size, hidden_dim, height, width)
        
        _, last_state_list = self.convlstm(x) # last_state_list: [batch, hidden_dim, H, W]
        residuals = self.output_layer(last_state_list[0][0].permute(0,2,3,1)) # residuals: [Batch, H, W, seqlen]
        last_twsa = x[:, -1, 0] if self.residual_from_last else torch.zeros_like(x[:, -1, 0]) # [batch, H, W]
        new_twsa_seq = []
        for res in residuals.permute(3, 0, 1, 2): # get sequence lemgth first  # [ Seqlen, Batch, H, W]
            last_twsa = last_twsa + res
            new_twsa_seq.append(last_twsa)
        pred = torch.stack(new_twsa_seq).permute(1, 0, 2, 3) # [Batch, Seqlen, H, W]
        
        
        return pred   
    
    def training_step(self, batch, batch_idx):

        x, y = torch.Tensor(batch['x_sequence']), torch.Tensor(batch['y_timestep'])
        land_mask = torch.Tensor(batch['land_mask']).bool()        
        output = self(x)
        loss = 0
        for month in range(self.horizon):
            loss_ = self.criterion(output[:, month][land_mask], y[:, month][land_mask]) 
            self.log(f'train_loss_month{month}', loss_, batch_size=x.shape[0],  on_epoch=True, on_step=False)
            loss += loss_ # sum the losses for each month     

        self.log('train_loss', loss)
        return loss
    
    def _reset_metrics(self):
        self.metrics['rmse'].reset()
        self.metrics['explained_var'].reset()

    def on_validation_epoch_start(self):
        self._reset_metrics()

    def on_test_epoch_start(self):
        self._reset_metrics()

    def validation_step(self, batch, batch_idx):

        x, y = torch.Tensor(batch['x_sequence']), torch.Tensor(batch['y_timestep'])

        land_mask = torch.Tensor(batch['land_mask']).bool()

        output = self(x)



        for name, metric in self.metrics.items():
            loss = 0
            for month in range(self.horizon):
                loss_ = metric(output[:,  month][land_mask], y[:, month][land_mask]) 
                self.log(f'val_{name}_month{month}', loss_, batch_size=x.shape[0],  on_epoch=True, on_step=False)
                loss += loss_ # sum the losses for each month
            if name == 'explained_var':
                loss = loss/self.horizon
            self.log(f'val_{name}', loss, batch_size=x.shape[0], on_epoch=True)


        
    def test_step(self, batch, batch_idx):
        x, y = torch.Tensor(batch['x_sequence']), torch.Tensor(batch['y_timestep'])
        land_mask = torch.Tensor(batch['land_mask']).bool()
        output = self(x)
        # output = output.squeeze(1)
        for name, metric in self.metrics.items():
            loss = 0
            for month in range(self.horizon):
                loss_ = metric(output[:,  month][land_mask], y[:, month][land_mask]) 
                self.log(f'test_{name}_month{month}', loss_, batch_size=x.shape[0],  on_epoch=True, on_step=False)
                loss += loss_ # sum the losses for each month
            if name == 'explained_var':
                loss = loss/self.horizon
            self.log(f'test_{name}', loss, batch_size=x.shape[0], on_epoch=True)
    
    def predict_step(self, batch, batch_idx):
        
        x, y = torch.Tensor(batch['x_sequence']), torch.Tensor(batch['y_timestep'])

        output = self(x) # output: [B, SeqLen, H, W]
        output = self._twsa_unnormalizer(output)
        y = self._twsa_unnormalizer(y)
        if 'twsa_offset' in batch: # anomaly mode: add C of the target months back -> absolute cm
            output = output + batch['twsa_offset']
            y = y + batch['twsa_offset']
        return output, y, x # Return the prediction, true value and the last input
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                 lr=self.learning_rate, weight_decay=1e-2)
        # return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.lr_scheduler is None:
            return optimizer
        if self.lr_scheduler != 'cosine':
            raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler}")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=self.learning_rate * 0.01)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
    
    def _twsa_unnormalizer(self, data):
        
        if self.trainer is not None and self.trainer.datamodule is not None:
            return self.trainer.datamodule.unnormalize_twsa(data)
        else:
            warnings.warn('No DataModule found.')
            return data
        

    def model_summary(self):
        """
        Print the model summary.
        """
        
        from pytorch_lightning.utilities.model_summary import ModelSummary
        print(ModelSummary(self))

# Example usage
# model = ConvLSTMLightningModule(input_dim=64, hidden_dim=[16, 32], kernel_size=(3, 3), num_layers=2)

# Example usage
# Create a PyTorch Lightning Trainer and fit your model
# trainer = pl.Trainer(max_epochs=10)
# model = ConvLSTMLightningModule(input_dim=3, hidden_dim=[64, 64], kernel_size=(3, 3), num_layers=2)
# trainer.fit(model, train_dataloder, val_dataloder)