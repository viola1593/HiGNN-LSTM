import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics
import tsl
# from tsl.metrics.torch.metrics import MaskedMAPE
from torchmetrics.regression import MeanAbsolutePercentageError
from losses.kge import KGE
from losses.utils_for_nans import mean_with_nans
from tsl.ops.connectivity import adj_to_edge_index



class Baseline_returnlast(pl.LightningModule):
    """Forward pass just returns the last time step of the input sequence."""
    def __init__(self, learning_rate=1e-3, 
                        data_image_shape=(36, 72),
                        land_sea_mask=None):
                      
        super(Baseline_returnlast, self).__init__()
        
        # data specific parameters
        self.data_image_shape = data_image_shape
        self.lsm = land_sea_mask
        if self.lsm is not None:
            self.lsm = torch.Tensor(self.lsm)
        self.model = nn.Linear(16,1)
        

 

        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        self.metrics = {'mape' : MeanAbsolutePercentageError(), 
                        'rmse' : nn.MSELoss(), 
                        'ssim' : torchmetrics.image.ssim.StructuralSimilarityIndexMeasure()}
        
        self.validation_step_outputs = []
        
    def forward(self, x):
        """
        x: [batches steps nodes features]
        """
        pred = self.model(x) # twsa should be the first feature
        
        return pred  
    
    def training_step(self, batch, batch_idx):
        
        x, y = torch.Tensor(batch['x_sequence']), torch.Tensor(batch['y_timestep'])
   
        output = self(x)
        output = output[:,-1,:].squeeze()
        
        loss = self.criterion(output, y.squeeze())
        self.log('train_loss', loss)
        return loss
    
    
    
    def validation_step(self, batch, batch_idx):
        x, y = torch.Tensor(batch['x_sequence']), torch.Tensor(batch['y_timestep'])
        
        output = x[:, -1, :, 0]
        output = output.squeeze()
        
        for metric_name, metric in self.metrics.items():
            metric = metric.to(output.device)
            if metric_name == 'ssim':
                loss = metric(output.reshape(output.shape[0], self.data_image_shape[0], self.data_image_shape[1]).unsqueeze(1), 
                              y.reshape(output.shape[0], self.data_image_shape[0], self.data_image_shape[1]).unsqueeze(1)) # 36 72 is the shape of the image
            else:
                loss = metric(output, y)
            if metric_name == 'rmse':
                loss = torch.sqrt(loss)
            
            self.log(f'val_{metric_name}', loss)
        self.validation_step_outputs.append((output, y))
        # loss = self.criterion(output, y)
        # self.log('val_loss', loss)

    def on_validation_start(self):
        self.validation_step_outputs.clear()
    def on_validation_end(self):
        '''this evaluation happens at the end of the validation to get the temporal correlations
        for the entire validation period for each pixel (although we only log the mean)'''
        predictions, targets = zip(*self.validation_step_outputs)
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        
        kge = KGE(predictions, targets, self.lsm) # KGE only makes sense for land pixels -> mask out sea pixels otherwise the mean will always be 0 or inf
        self.logger.experiment.log({'mean_val_kge': mean_with_nans(kge[0])}) # normal logging throws an error
        self.logger.experiment.log({'mean_val_corr': mean_with_nans(kge[1])})
        
        self.logger.experiment.log({'mean_val_alpha': mean_with_nans(kge[2])})
        self.logger.experiment.log({'mean_val_beta': mean_with_nans(kge[3])})



        
        
    def test_step(self, batch, batch_idx):
        x, y = torch.Tensor(batch['x_sequence']), torch.Tensor(batch['y_timestep'])
        
        output = x[:, -1, :, 0]
        output = output.squeeze()

        for metric_name, metric in self.metrics.items():
            metric = metric.to(output.device)
            if metric_name == 'ssim':
                loss = metric(output.reshape(output.shape[0], self.data_image_shape[0], self.data_image_shape[1]).unsqueeze(1), 
                              y.reshape(output.shape[0], self.data_image_shape[0], self.data_image_shape[1]).unsqueeze(1))
            else:
                loss = metric(output, y)
            if metric_name == 'rmse':
                loss = torch.sqrt(loss)
            
            self.log(f'val_{metric_name}', loss)
        # loss = self.criterion(output, y)
        # self.log('test_loss', loss)
    
    def predict_step(self, batch, batch_idx):
        
        x, y = torch.Tensor(batch['x_sequence']), torch.Tensor(batch['y_timestep'])
        
        output = x[:, -1, :, 0]
        output = output.squeeze()
        
        return output, y.squeeze(), x # Return the prediction, true value and the last input
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# Example usage
# model = ConvLSTMLightningModule(input_dim=64, hidden_dim=[16, 32], kernel_size=(3, 3), num_layers=2)

# Example usage
# Create a PyTorch Lightning Trainer and fit your model
# trainer = pl.Trainer(max_epochs=10)
# model = ConvLSTMLightningModule(input_dim=3, hidden_dim=[64, 64], kernel_size=(3, 3), num_layers=2)
# trainer.fit(model, train_dataloder, val_dataloder)