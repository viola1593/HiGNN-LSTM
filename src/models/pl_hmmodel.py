import warnings

import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils import to_dense_adj
from torchmetrics.regression import MeanSquaredError

from models.hierarchical_model import HiGNN_LSTM
from utils.utils import plot_graph


class HMModule(pl.LightningModule):
    def __init__(self, num_input_features, 
                        num_ocean_features, 
                        sequence_length, 
                        output_seq_len,
                        timestep_embed_dim,
                        learning_rate=1e-3,
                        lr_scheduler=None,
                        hidden_dim=128,
                        num_processing_gnns=3,
                        gnn_embedding_dim=4,
                        skip_gnn=False,
                        rnn_layers=3,
                        rnn_hidden_size=32,
                        rnn_dropout=0.0,
                        land_sea_mask=None,
                        plot_intermediate=False,
                        use_ocean_nodes=True,
                        aggregation='sum',
                        residual_from_last=True,
                        **kwargs
                        
            ):
        
        """
         
                Args:
                    ""num_input_features (int): Number of input features.
                    num_ocean_features (int): Number of ocean features.
                    sequence_length (int): Length of the input sequence.
                    output_seq_len (int): Number of timesteps to forecast aka forceasting horizon.
                    timestep_embed_dim (int): Dimension of the timestep embedding. Required, the
                        model is always conditioned on the forecasting timestep.
                    learning_rate (float, optional): Learning rate. Default is 1e-3.
                    lr_scheduler (str, optional): None or 'cosine' (annealed to 1% of learning_rate
                        over trainer.max_epochs). Default is None.
                    hidden_dim (int, optional): Hidden dimension for the GNN layers. Default is 128.
                    num_processing_gnns (int, optional): Number of GNN layers in the processing step. Default is 3.
                    gnn_embedding_dim (int, optional): Dimension of the output from GNN. Default is 4.
                    rnn_layers (int, optional): Number of RNN layers. Default is 3.
                    rnn_hidden_size (int, optional): Hidden size for the RNN layers. Default is 32.
                    rnn_dropout (float, optional): Dropout for the RNN layers. Default is 0.0.
                    land_sea_mask (np.array, optional): Land-sea mask. Default is None.
                    use_ocean_nodes (bool, optional): Whether the graph contains ocean nodes. Must match the
                        graph variant of the datamodule (False for the no_ocean variant). Default is True.
                    aggregation (str, optional): 'sum' or 'mean' message reduction in the GNN layers. Default is 'sum'.
                    residual_from_last (bool, optional): Whether the forecast is a residual on top of the last
                        input TWSA. False predicts TWSA directly (e.g. anomalies). Default is True.
                    **kwargs: Additional keyword arguments.
                 
                   """""
        super(HMModule, self).__init__()
        self.num_features = num_input_features
        self.num_ocean_features = num_ocean_features
        self.sequence_len = sequence_length
        self.horizon = output_seq_len
        self.plot_intermediate = plot_intermediate
        self.lsm = land_sea_mask
        if self.lsm is not None:
            self.lsm = torch.Tensor(self.lsm)
        
        
        
        self.model = HiGNN_LSTM(num_features=self.num_features,num_ocean_features=self.num_ocean_features,
                                        horizon=self.horizon,  
                                        sequence_length=self.sequence_len, twsa_index=0, 
                                        hidden_dim=hidden_dim, 
                                        num_processing_gnns=num_processing_gnns, 
                                        skip_gnn=skip_gnn,
                                        gnn_embedding_dim=gnn_embedding_dim,
                                        rnn_hidden_size=rnn_hidden_size, 
                                        rnn_layers=rnn_layers, 
                                        rnn_dropout=rnn_dropout,
                                        aggregation=aggregation,
                                        timestep_embed_dim=timestep_embed_dim,
                                        use_ocean_nodes=use_ocean_nodes,
                                        residual_from_last=residual_from_last,
                                                        )
 


        self.num_nodes = None
        self.antarctica_mask = None
        self.lat = None
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler




        self.metrics = nn.ModuleDict({'mse' : nn.MSELoss(),
                        'rmse': MeanSquaredError(squared=False), # RMSE
                        })

        self.prediction_gnn_outputs = None
        
        
    def forward(self, data):
        """
        x: [batches steps nodes features]
        """
        # x: [batches steps nodes features]
         
        # pred: [batches steps nodes features=1] --> [batches steps nodes]
        
        pred, heterograph, gnn_outputs = self.model(data)

        return pred, heterograph, gnn_outputs 
     
    def on_train_start(self):
        self.num_nodes = self.trainer.datamodule.data_train.num_land_nodes()
        self.ocean_clusters = self.trainer.datamodule.num_ocean_clusters
        if self.antarctica_mask is None:
            self.antarctica_mask = self._get_relevant_land().to(next(self.parameters()).device).bool() # shape [num nodes]
        if self.lat is None:
            self.lat = torch.Tensor(self.trainer.datamodule.data_train.land[0][:,1]).to(next(self.parameters()).device)
        self.lat_weights = self._get_lat_weight()


    def training_step(self, batch, batch_idx):
        output, _, _ = self(batch) # output shape: [batch_size* num nodes, horizon]
        
        # get batch size 
        batch_size = batch.batch_size
        
        output = output.reshape(batch_size, self.num_nodes, self.horizon)[:, self.antarctica_mask] # reshape to [batch_size, num nodes, horizon]
        gt = batch['grid'].y.reshape(batch_size, self.num_nodes, self.horizon)[:, self.antarctica_mask] # reshape to [batch_size, num nodes, horizon] # -20 to ignore the sea pixels
        
        loss = torch.abs(output - gt)
        
        loss = (loss * self.lat_weights.unsqueeze(0).unsqueeze(-1)) # [batch, nodes, horizon]
        loss = loss.sum(dim=[1,2]).mean()

        self.log('train_loss', loss, batch_size=output.shape[0], on_epoch=True)

        return loss
    
    def on_validation_start(self):
        self.num_nodes = self.trainer.datamodule.data_val.num_land_nodes()
        self.ocean_clusters = self.trainer.datamodule.num_ocean_clusters
        if self.antarctica_mask is None:
            self.antarctica_mask = self._get_relevant_land().to(next(self.parameters()).device).bool()  # shape [num nodes]

    def on_validation_epoch_start(self):
        self.metrics['rmse'].reset()

    def validation_step(self, batch, batch_idx):
        
        output, graphs, _ = self(batch)        
        # get batch size 
        batch_size = batch.batch_size
        
        # save for later plotting
        self.last_validation_step_graph = graphs
        self.last_validation_step_output = output.detach()
        self.last_validation_step_target = batch['grid'].y.detach()

        
        output = output.reshape(batch_size, self.num_nodes, self.horizon)[:, self.antarctica_mask] # reshape to [batch_size, num nodes, horizon]
        gt = batch['grid'].y.reshape(batch_size, self.num_nodes, self.horizon)[:, self.antarctica_mask] # reshape to [batch_size, num nodes, horizon] # -20 to ignore the sea pixels
        loss_mse=0
        loss_rmse=0
        for month in range(self.horizon):
            loss_ = self.metrics['mse'](output[:, :, month], gt[:, :, month]) 
            self.log(f'val_mse_month{month}', loss_, batch_size=output.shape[0],  on_epoch=True, on_step=False)
            loss_mse += loss_ # sum the losses for each month
            loss_ = self.metrics['rmse'](output[:, :, month], gt[:, :, month])
            self.log(f'val_rmse_month{month}', loss_, batch_size=output.shape[0],  on_epoch=True, on_step=False)
            loss_rmse += loss_ #
        self.log(f'val_mse', loss_mse, batch_size=output.shape[0], on_epoch=True)
        self.log(f'val_rmse', loss_rmse, batch_size=output.shape[0], on_epoch=True)


    def on_validation_end(self):
        '''Plotting intermediate TWSA predictions and graphs to wandb.
        TWSA plots show the first prediction in the last batch of the validation set.
        Corresponding mesh2mesh adjacency matrix is plotted for each month.'''
        if self.plot_intermediate:
            if (self.trainer.current_epoch % self.plot_intermediate)==0:
                                
                predictions = self.last_validation_step_output.reshape(-1, self.num_nodes, self.horizon)
                targets = self.last_validation_step_target.reshape(-1, self.num_nodes, self.horizon)
                # in anomaly mode (datamodule.twsa_anomaly) these maps show twsa - C, not absolute TWSA
                pred = self._twsa_unnormalizer(predictions[0]) # get the prediction of the first batch shape [batch_size,num nodes, horizon]
                true = self._twsa_unnormalizer(targets[0]) # get the ground truth of the first batch shape [batch_size,num nodes, horizon]
              
                for i in range(self.horizon):
                    twsa_pred = pred[:,i].float()
                    twsa_true = true[:,i].squeeze() 
                    self.plot_predictions(twsa_pred, twsa_true, month=i)

      

    
    def on_test_start(self):
        self.num_nodes = self.trainer.datamodule.data_test.num_land_nodes()
        self.ocean_clusters = self.trainer.datamodule.num_ocean_clusters
        self.antarctica_mask = self._get_relevant_land().to(next(self.parameters()).device).bool() # shape [num nodes]

    def on_test_epoch_start(self):
        self.metrics['rmse'].reset()

    def test_step(self, batch, batch_idx):
        output, _, _ = self(batch)        
        batch_size = batch.batch_size


        output = output.reshape(batch_size, self.num_nodes, self.horizon)[:, self.antarctica_mask] # reshape to [batch_size, num nodes, horizon]
        gt = batch['grid'].y.reshape(batch_size, self.num_nodes, self.horizon)[:, self.antarctica_mask] # reshape to [batch_size, num nodes, horizon] # -20 to ignore the sea pixels

        loss_mse=0
        loss_rmse=0
        for month in range(self.horizon):
            loss_ = self.metrics['mse'](output[:, :, month], gt[:, :, month]) 
            self.log(f'test_mse_month{month}', loss_, batch_size=output.shape[0],  on_epoch=True, on_step=False)
            loss_mse += loss_ # sum the losses for each month
            loss_ = self.metrics['rmse'](output[:, :, month], gt[:, :, month])
            self.log(f'test_rmse_month{month}', loss_, batch_size=output.shape[0],  on_epoch=True, on_step=False)
            loss_rmse += loss_ #
        self.log(f'test_mse', loss_mse, batch_size=output.shape[0], on_epoch=True)
        self.log(f'test_rmse', loss_rmse, batch_size=output.shape[0], on_epoch=True)


    
    
    def predict_step(self, batch, batch_idx):
        """
        This method is called when using the predict method of the trainer.
        It returns the predictions with shape [batch_size*nodes, horizon].
        """
        output, heterographs, gnn_outputs = self(batch)
        self.prediction_gnn_outputs = gnn_outputs # save the last gnn outputs for visualization
        output = self._twsa_unnormalizer(output)
        y = self._twsa_unnormalizer(batch['grid'].y)
        if 'twsa_offset' in batch['grid']: # anomaly mode: add C of the target months back -> absolute cm
            output = output + batch['grid'].twsa_offset
            y = y + batch['grid'].twsa_offset
        return output, y, batch, heterographs, gnn_outputs # Return the prediction, true value and the last input
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                 lr=self.learning_rate, weight_decay=1e-2)
        if self.lr_scheduler is None:
            return optimizer
        if self.lr_scheduler != 'cosine':
            raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler}")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=self.learning_rate * 0.01)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
    
    def _get_lat_weight(self):
         # compute latitude weights for the land pixels (apply antarctica mask to match later indexing)
        lat_rad = self.lat * np.pi / 180.0
        lat_w = torch.cos(lat_rad).to(next(self.parameters()).device).float()
        # keep only the land nodes used later (after masking out Antarctica)
        self.lat_weights = lat_w[self.antarctica_mask].clone()
        # normalize so the weights sum to 1 (avoid changing overall loss scale)
        self.lat_weights = self.lat_weights / (self.lat_weights.sum() + 1e-12)
        return self.lat_weights
    

    
    def _twsa_unnormalizer(self, data):
        if self.trainer is not None and self.trainer.datamodule is not None:
            return self.trainer.datamodule.unnormalize_twsa(data)
        else:
            warnings.warn('No DataModule found.')
            return data
        
    def _get_relevant_land(self, greenland=True):
        '''Retrieve and flip a mask where pixels of Antarctica == True from the datamodule.
        if greenland is False, also mask out Greenland.
        This masked can be used on the land coordinates only. 
        Returns:
            torch.Tensor: A boolean tensor where True indicates land pixels that are not in Antarctica.
        '''
        if self.trainer is not None and self.trainer.datamodule is not None:
            if greenland:
                return torch.Tensor(~self.trainer.datamodule.get_antarctica_landmask())
            else:
                ant = self.trainer.datamodule.get_antarctica_landmask()
                gl = self.trainer.datamodule.get_greenland_landmask()
                mask = ~(ant | gl)
                return torch.Tensor(mask)
        else:
            warnings.warn('No DataModule found.')
            return None
        
## A few plotting functions 
    def plot_predictions(self, predictions, targets, month=0):
        """
        Draw the predictions and targets for a given batch.
        """
        
        # Convert tensors to numpy arrays
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        
        lon, lat = self.trainer.datamodule.data_val.land[0][:,0], self.trainer.datamodule.data_val.land[0][:,1]
        
        vmax = 15
        f, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(5,10))
        p1 = ax1.scatter(lon, lat, c=predictions, cmap='coolwarm', s=1,vmin=-vmax, vmax=vmax)
        p2 = ax2.scatter(lon, lat, c=targets, cmap='coolwarm', s=1,vmin=-vmax, vmax=vmax)
        ape = (predictions-targets)/(targets + 1e-8) * 100  # Avoid division by zero
        p3 = ax3.scatter(lon, lat, c=ape, cmap='RdBu',s=1, vmax=100, vmin=-100)
        
        ax1.set_title('Predicted TWSA')
        ax2.set_title('True TWSA')
        ax3.set_title('Percentage Difference')
        epoch = self.trainer.current_epoch
        f.suptitle(f'TWSA month={month} (Val Set)')
        
        plt.tight_layout()
        plt.colorbar(p1, ax=ax1, extend='both')
        plt.colorbar(p2, ax=ax2, extend='both')
        plt.colorbar(p3, ax=ax3, extend='both')

        if self.trainer is not None:
            self.trainer.logger.experiment.log({f'Validation_month_{month}': wandb.Image(plt),
                                                'epoch': epoch,
                                                'month': month,
                                                'global_step': self.global_step})
            plt.close() 

    def plot_graph(self, edge_index, edge_attr, pos_lat_lon, title=None, latlon_dest=None, vmax=None):
        """
        This will plot the graph 
        """
        fig, axis = plot_graph( edge_index, pos_lat_lon, title, latlon_dest, vmax, edge_attr)  # W,H)
        if self.trainer is not None:
            self.trainer.logger.experiment.log({title: wandb.Image(fig),
                                                'epoch': self.trainer.current_epoch,
                                                'global_step': self.global_step})
            plt.close()

    def plot_intermediate_adjacency_from_edges(self, graph, num_m2m_edges, month):
        edges = graph.edge_index[:,:num_m2m_edges]
        # edge_attr are of shape [num_edges, hidden_dim], so we take the sum
        edge_attr = torch.abs(graph.edge_attr[:num_m2m_edges]).sum(dim=1)
        if self.trainer is not None:  
            adj = to_dense_adj(edges, edge_attr=edge_attr)[0].cpu()
            plt.imshow(adj, cmap='magma_r', interpolation='none')
            # plt.imshow(adj, cmap='RdBu',norm=matplotlib.colors.TwoSlopeNorm(0), interpolation='none')
            plt.colorbar()
            plt.title(f'Mesh2Mesh Summed Adjacency Month {month+1}')
        
            self.trainer.logger.experiment.log({f'Adj month {month+1}': wandb.Image(plt),
                                        'epoch': self.trainer.current_epoch,
                                        'global_step': self.global_step})
            plt.close()

    def model_summary(self):
        """
        Print the model summary.
        """
        from torchinfo import summary
        print(summary(self.model, verbose=0, depth=2))
