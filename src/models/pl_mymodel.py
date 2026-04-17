import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics
import tsl
# from tsl.metrics.torch.metrics import MaskedMAPE
from torchmetrics.regression import MeanAbsolutePercentageError
from losses.kge import KGE
from models.mymodel import TWSA_forecaster
from losses.utils_for_nans import mean_with_nans, mean_abs_with_nans
from tsl.ops.connectivity import adj_to_edge_index
from torch_geometric.utils import to_dense_adj




from tsl.nn.models.stgn.rnn_gcn_model import RNNEncGCNDecModel

class myModule(pl.LightningModule):
    def __init__(self, num_input_features, sequence_length, output_seq_len, 
                        num_months_per_gnn=None,
                        learning_rate=1e-3, 
                        
                        land_sea_mask=None,
                        **kwargs
                        
            ):
        
        """
         
                Args:
                    ""num_input_features (int): Number of input features.
                    output_seq_len (int): Number of timesteps to forecast aka forceasting horizon.
                    learning_rate (float, optional): Learning rate. Default is 1e-3.
                    land_sea_mask (np.array, optional): Land-sea mask. Default is None.
                    **kwargs: Additional keyword arguments.
                 
                   """""
        super(myModule, self).__init__()
        self.num_features = num_input_features
        self.sequence_len = sequence_length
        self.horizon = output_seq_len
        self.lsm = land_sea_mask
        if self.lsm is not None:
            self.lsm = torch.Tensor(self.lsm)
        
        
        self.model = TWSA_forecaster(self.num_features, horizon=self.horizon, 
                                     sequence_length=self.sequence_len, twsa_index=0,
                                    num_months_per_gnn=num_months_per_gnn
                                    )


        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        self.metrics = {#'mape' : MeanAbsolutePercentageError(), mape always really bad because low values make high errors
                        'rmse' : nn.MSELoss(), 
                        # 'ssim' : torchmetrics.image.ssim.StructuralSimilarityIndexMeasure()
                        }
        
        self.adj = None
        self.twsa_scaling_mean_std = (torch.Tensor(0), torch.Tensor(1))
        self.validation_step_outputs = []
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        x: [batches steps nodes features]
        """
        # x: [batches steps nodes features]
         
        # pred: [batches steps nodes features=1] --> [batches steps nodes]
        pred = self.model(x, edge_index).squeeze(dim=-1)
        # Instead of adding only the last timestep, use a weighted combination
        # import pdb; pdb.set_trace()
        # last_value = x[:,-1,:,0].unsqueeze(1)
        # alpha = torch.sigmoid(self.alpha_param)  # learnable parameter between 0 and 1
        # pred = alpha * last_value + (1 - alpha) * res
        return pred  
    
    def training_step(self, batch, batch_idx):
        
        x, y = torch.Tensor(batch['x_sequence']), torch.Tensor(batch['y_timestep'])
        # print('x min/max', x[:,:,:,0].min(), x[:,:,:,0].max())
        # print('x mean/std',x[:,:,:,0].mean(), x[:,:,:,0].std())
        # print('y min/max', y.min(), y.max())
        # print('y mean/std',y.mean(), y.std())
        edge_index = torch.Tensor(batch['edge_index'][0]).long()
        edge_weight = torch.ones(edge_index.shape[1]).to(x.device)
        output = self(x, edge_index, edge_weight)
        
        # output shape: [batch, seq_len, num_nodes, num_features=1]
        loss=0
        for i in range(self.horizon):
            
            loss += self.criterion(output[:,i,:-20], y[:,i,:-20]) # -20 to ignore the sea pixels
        self.log('train_loss', loss)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        
        x, y = torch.Tensor(batch['x_sequence']), torch.Tensor(batch['y_timestep'])
        edge_index = torch.Tensor(batch['edge_index'][0]).long()
        edge_weight = torch.ones(edge_index.shape[1]).to(x.device)
        output = self(x,edge_index, edge_weight)
        loss=0
        for i in range(self.horizon):
            loss += self.criterion(output[:,i,:-20], y[:,i,:-20]) # -20 to ignore the sea pixels
        self.log('val_loss', loss)
        # mean, std = self.twsa_scaling_mean_std[0].to(x.device), self.twsa_scaling_mean_std[1].to(x.device)
        # output = output[:,:,:-20]*std + mean
        # y = y[:,:,:-20]*std + mean
        
        antarctica = self._get_antarctica().to(x.device)
        output = self._data_unnormalizer(output[:,:,:-20])*antarctica
        y = self._data_unnormalizer(y[:,:,:-20])*antarctica
        
        # import pdb; pdb.set_trace()
        for metric_name, metric in self.metrics.items():
            metric = metric.to(output.device)
            loss = metric(output, y)
            if metric_name == 'rmse':
                loss = torch.sqrt(loss)
            
            self.log(f'val_{metric_name}', loss)

        # metric per month
        for i in range(self.horizon):
            for metric_name, metric in self.metrics.items():
                metric = metric.to(output.device)
                loss = metric(output[:,i], y[:,i])
                if metric_name == 'rmse':
                    loss = torch.sqrt(loss)
                
                self.log(f'val_{metric_name}_month_{i}', loss)

        # calculate KGE for each predicted sequence and per pixel
        kge_list = []
        # for every element in the batch
        for pred, true in zip(output, y):
            kge = KGE(pred, true, self.lsm)
            kge_list.append(kge)
        #calculate the mean KGE over the batch
        pixel_kge = torch.stack(kge_list).mean(axis=0)
        self.log('val_kge_batchmean', mean_with_nans(pixel_kge[0])) # mean over all pixels
        self.log('val_corr_batchmean', mean_with_nans(pixel_kge[1]))
        self.log('val_alpha_batchmean', mean_with_nans(pixel_kge[2]))
        self.log('val_beta_batchmean', mean_with_nans(pixel_kge[3]))

        self.validation_step_outputs.append((output, y)) # last 20 pixels are sea pixels
        # loss = self.criterion(output, y)
        # self.log('val_loss', loss)

    def on_validation_start(self):
        self.validation_step_outputs.clear()
        self.twsa_scaling_mean_std = self._get_data_scaler()
    def on_validation_end(self):
        '''this evaluation happens at the end of the validation to get the temporal correlations
        for the entire validation period for each pixel (although we only log the mean)'''
        pass
        # predictions, targets = zip(*self.validation_step_outputs)
        
        # predictions = torch.cat(predictions, dim=0)
        # targets = torch.cat(targets, dim=0)
        # twsa_mean, twsa_std = self.trainer.datamodule.scaling_mean_std[0]
        # predictions = predictions*torch.Tensor(twsa_std.values) + torch.Tensor(twsa_mean.values)
        # targets = targets*torch.Tensor(twsa_std.values) + torch.Tensor(twsa_mean.values)
        # # import pdb; pdb.set_trace()
        # kge = KGE(predictions, targets, self.lsm) # KGE only makes sense for land pixels -> mask out sea pixels otherwise the mean will always be 0 or inf
        
        # self.logger.experiment.log({'mean_val_kge': mean_with_nans(kge[0])}) # normal logging throws an error
        # self.logger.experiment.log({'mean_val_corr': mean_abs_with_nans(kge[1])}) # rmse with it self since correlation can also be negative
        
        # self.logger.experiment.log({'mean_val_alpha': mean_abs_with_nans(kge[2])})
        # self.logger.experiment.log({'mean_val_beta': mean_abs_with_nans(kge[3])})



        
        
    def test_step(self, batch, batch_idx):
        x, y = torch.Tensor(batch['x_sequence']), torch.Tensor(batch['y_timestep'])
        twsa_last = x[:,-1,:, 0].unsqueeze(1)
        edge_index = torch.Tensor(batch['edge_index'][0]).long()
        edge_weight = torch.ones(edge_index.shape[1]).to(x.device)
        output = self(x,edge_index, edge_weight)
        #output = pred_residue + twsa_last
        # mean, std = self.twsa_scaling_mean_std[0].to(x.device), self.twsa_scaling_mean_std[1].to(x.device)
        # output = output[:,:,:-20]*std + mean
        # y = y[:,:,:-20]*std + mean
        antarctica = self._get_antarctica().to(x.device)
        output = self._data_unnormalizer(output[:,:,:-20])*antarctica
        y = self._data_unnormalizer(y[:,:,:-20])*antarctica

        for metric_name, metric in self.metrics.items():
            metric = metric.to(output.device)
            loss = metric(output, y)
            if metric_name == 'rmse':
                loss = torch.sqrt(loss)
            
            self.log(f'val_{metric_name}', loss)


    def on_test_start(self):
        self.twsa_scaling_mean_std = self._get_data_scaler()
    
    
    def predict_step(self, batch, batch_idx):
        x, y = torch.Tensor(batch['x_sequence']), torch.Tensor(batch['y_timestep'])
        #twsa_last = x[:,-1,:, 0].unsqueeze(1)
        edge_index = torch.Tensor(batch['edge_index'][0]).long()
        edge_weight = torch.ones(edge_index.shape[1]).to(x.device)
        output = self(x,edge_index, edge_weight)
        #import pdb; pdb.set_trace()
        output = self._data_unnormalizer(output[:,:,:-20])
        y = self._data_unnormalizer(y[:,:,:-20])
    
        
        return output, y, x # Return the prediction, true value and the last input
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def _get_data_scaler(self):
        """Dynamically retrieve the target scaler from the DataModule."""
        if self.trainer is not None and self.trainer.datamodule is not None:
            
            if self.trainer.datamodule.scaling_mean_std[0][0].name == 'twsa':  # index 0 should be the twsa scaler
                mean, var = self.trainer.datamodule.scaling_mean_std[0]
                return torch.Tensor(mean.values), torch.Tensor(var.values)
            else: 
                Warning('TWSA scaler not found.')
        else: 
            Warning('No DataModule found.')
            return torch.Tensor(0), torch.Tensor(1)  # Handle case where scaler is not available
    
    def _data_unnormalizer(self, data):
        if self.trainer is not None and self.trainer.datamodule is not None:
            return self.trainer.datamodule.unnormalize_twsa(data)
        else: 
            Warning('No DataModule found.')
            return data
        
    def _get_antarctica(self):
        '''Retrieve a mask where pixels of Antarctica == True from the datamodule.
        This masked can be used on the land coordinates only. '''
        if self.trainer is not None and self.trainer.datamodule is not None:
            return torch.Tensor(~self.trainer.datamodule.get_antarctica_landmask())
        else: 
            Warning('No DataModule found.')
            return None

# Example usage
# model = ConvLSTMLightningModule(input_dim=64, hidden_dim=[16, 32], kernel_size=(3, 3), num_layers=2)

# Example usage
# Create a PyTorch Lightning Trainer and fit your model
# trainer = pl.Trainer(max_epochs=10)
# model = ConvLSTMLightningModule(input_dim=3, hidden_dim=[64, 64], kernel_size=(3, 3), num_layers=2)
# trainer.fit(model, train_dataloder, val_dataloder)