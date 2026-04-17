import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics
import tsl
import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric as pyg
import matplotlib
from torch_geometric.utils import to_dense_adj
# from tsl.metrics.torch.metrics import MaskedMAPE
from torchmetrics.regression import MeanSquaredError
from losses.kge import KGE
from models.hierarchical_model import HM_forecaster_auto, HM_forecaster_seq2seq
from models.layers.mlp import MLP
from utils.utils import plot_graph


class HMModule(pl.LightningModule):
    def __init__(self, num_input_features, num_ocean_features, sequence_length, output_seq_len, 
                        forecasting_method='autoregressive',
                        learning_rate=1e-3, 
                        hidden_dim=128, 
                        num_processing_gnns=3,
                        gnn_embedding_dim=4,
                        rnn_layers=3,
                        rnn_hidden_size=32,
                        rnn_dropout=0.0,
                        seq_emb_length=4,
                        frac_teacher_forcing_epochs=0.5,
                        timestep_embed_dim=None,
                        land_sea_mask=None,
                        plot_intermediate=False,
                        gnn_method='simple',
                        num_heads=8,
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
        super(HMModule, self).__init__()
        self.num_features = num_input_features
        self.num_ocean_features = num_ocean_features
        self.sequence_len = sequence_length
        self.horizon = output_seq_len
        self.plot_intermediate = plot_intermediate
        self.lsm = land_sea_mask
        if self.lsm is not None:
            self.lsm = torch.Tensor(self.lsm)
        
        self.gnn_method = gnn_method
        self.forecasting_method = forecasting_method  # or autoregressive
        if self.forecasting_method == 'autoregressive':
        
            self.model = HM_forecaster_auto(num_features=self.num_features,num_ocean_features=self.num_ocean_features,
                                            horizon=self.horizon,  
                        sequence_length=self.sequence_len, twsa_index=0, 
                        hidden_dim=hidden_dim, 
                        num_processing_gnns=num_processing_gnns, 
                        gnn_embedding_dim=gnn_embedding_dim,
                        rnn_hidden_size=rnn_hidden_size, 
                        rnn_layers=rnn_layers, 
                        rnn_dropout=rnn_dropout,
                        seq_emb_length=seq_emb_length,
                        timestep_embed_dim=timestep_embed_dim, 
                        method=self.gnn_method,
                        num_heads=num_heads
                                        )
        elif self.forecasting_method == 'seq2seq':
            self.model = HM_forecaster_seq2seq(num_features=self.num_features, 
                                            horizon=self.horizon,  
                        sequence_length=self.sequence_len, twsa_index=0, 
                        hidden_dim=hidden_dim, 
                        num_processing_gnns=num_processing_gnns, 
                        gnn_embedding_dim=gnn_embedding_dim,
                        rnn_hidden_size=rnn_hidden_size, 
                        rnn_layers=rnn_layers, 
                        rnn_dropout=rnn_dropout,
                        seq_emb_length=seq_emb_length,
                        timestep_embed_dim=timestep_embed_dim, 
                        method=self.gnn_method,
                        num_heads=num_heads
                                        )
        else: 
            raise ValueError(f'Unknown forecasting method {self.forecasting_method}. Choose either "autoregressive" or "seq2seq".')

        self.num_nodes = None
        self.antarctica_mask = None
        self.lat = None
        self.learning_rate = learning_rate
        self.frac_teacher_forcing_epochs = frac_teacher_forcing_epochs

        # self.aux_mlp = MLP(in_dim=self.sequence_len*gnn_embedding_dim, 
        #                    hid_dim=16, out_dim=1, last=True)

        self.criterion = nn.MSELoss() 
        self.metrics = {#'loss': self.criterion,
                        'mse' : nn.MSELoss(),
                        'rmse': MeanSquaredError(squared=False), # RMSE
                        'explained_var': torchmetrics.ExplainedVariance(),  
                        # 'ssim' : torchmetrics.image.ssim.StructuralSimilarityIndexMeasure()
                        }
        
        self.log_graph_edge_attr = False
        self.prediction_gnn_outputs = None
        self.attention = None
        
        
    def forward(self, data, teacher_forcing_ratio=0.0):
        """
        x: [batches steps nodes features]
        """
        # x: [batches steps nodes features]
         
        # pred: [batches steps nodes features=1] --> [batches steps nodes]
        
        pred, heterograph, gnn_outputs = self.model(data,
                                       teacher_forcing_ratio=teacher_forcing_ratio)

        return pred, heterograph, gnn_outputs 
     
    def on_train_start(self):
        self.num_nodes = self.trainer.datamodule.data_train.num_land_nodes()
        self.ocean_clusters = self.trainer.datamodule.num_ocean_clusters
        if self.antarctica_mask is None:
            self.antarctica_mask = self._get_relevant_land().to(next(self.parameters()).device).bool() # shape [num nodes]
        if self.lat is None:
            self.lat = torch.Tensor(self.trainer.datamodule.data_train.land[0][:,1]).to(next(self.parameters()).device)
        self.lat_weights = self._get_lat_weight()
        self.monthly_loss_weights = torch.Tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
    def adapt_monthly_loss_weights(self):
        zero_indices = (self.monthly_loss_weights == 0).nonzero(as_tuple=True)[0]
        if len(zero_indices) > 0:
            self.monthly_loss_weights[zero_indices[0]] = 1

    def training_step(self, batch, batch_idx):
        # Calculate teacher forcing ratio that decreases over time for scheduled sampling
        if self.frac_teacher_forcing_epochs > 0.:
            teacher_forcing_ratio = max(0.0005, 1.0 - (self.trainer.current_epoch / (self.trainer.max_epochs * self.frac_teacher_forcing_epochs)))
        else: teacher_forcing_ratio=0.0
        output, _, gnn_features = self(batch, teacher_forcing_ratio=teacher_forcing_ratio) # output shape: [batch_size* num nodes, horizon]
        
        # get batch size 
        batch_size = batch.batch_size

        
        output = output.reshape(batch_size, self.num_nodes, self.horizon)[:, self.antarctica_mask] # reshape to [batch_size, num nodes, horizon]
        gt = batch['grid'].y.reshape(batch_size, self.num_nodes, self.horizon)[:, self.antarctica_mask] # reshape to [batch_size, num nodes, horizon] # -20 to ignore the sea pixels
        # last_twsa = batch['grid'].x[:, -1, 0].reshape(batch_size, self.num_nodes)[:, self.antarctica_mask]  # shape [batch_size, num nodes]
        
        loss = 0  
        # if self.trainer.current_epoch % 10==0:
        #     self.adapt_monthly_loss_weights()
        # residual = 0
        # calculate loss per month
        loss = torch.abs(output - gt)
        
        loss = (loss * self.lat_weights.unsqueeze(0).unsqueeze(-1)) # [batch, nodes, horizon]
        loss = loss.sum(dim=[1,2]).mean()

        # for month in range(self.horizon):
        #     se = torch.abs((output[:, :, month] - gt[:, :, month]))           # [batch, nodes]
        #     # weighted MAE: lat-weighted mean over batches, then sum over nodes
        #     sample_mse = (se * self.lat_weights.unsqueeze(0)).mean(dim=0)  # [nodes]
        #     loss_ = sample_mse.sum() 
            # # aux loss on gnn features
            # gnn_features_month = gnn_features[month].reshape(batch_size* self.num_nodes, -1)  # shape [batch_size*num nodes, sequence_len, gnn_embedding_dim]
            # aux_pred = self.aux_mlp(gnn_features_month)
            # aux_loss_ = self.criterion(aux_pred.squeeze(), batch['grid'].y[:,month]) 
            
        #     # self.log(f'aux_loss_month{month}', aux_loss_, batch_size=output.shape[0],  on_epoch=True, on_step=False)
        #     self.log(f'train_loss_month{month}', loss_, batch_size=output.shape[0],  on_epoch=True, on_step=False)
        #     loss = loss + loss_ #* self.monthly_loss_weights[month] # sum the losses for each month     

            # loss on residuals
            # residual += ((output[:, :, month] - last_twsa).sum(dim=1) -(gt[:, :, month] - last_twsa).sum(dim=1)).abs().mean()
            # last_twsa = output[:, :, month]  # update last_twsa for next month
            # self.log(f'res_loss_month{month}', residual, batch_size=output.shape[0],  on_epoch=True, on_step=False)

        #self.log('gt_forcing_rationg', teacher_forcing_ratio, batch_size=batch_size, on_epoch=True)
        self.log('train_loss', loss, batch_size=output.shape[0], on_epoch=True)
        # self.log('res_loss', residual, batch_size=output.shape[0], on_epoch=True)
        
        return loss #+0.0001*residual
    
    def on_validation_start(self):
        self.num_nodes = self.trainer.datamodule.data_val.num_land_nodes()
        self.ocean_clusters = self.trainer.datamodule.num_ocean_clusters
        if self.antarctica_mask is None:
            self.antarctica_mask = self._get_relevant_land().to(next(self.parameters()).device).bool()  # shape [num nodes]
   
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
            _rmse = self.metrics['rmse'].to(gt.device)
            loss_ = _rmse(output[:, :, month], gt[:, :, month])
            self.log(f'val_rmse_month{month}', loss_, batch_size=output.shape[0],  on_epoch=True, on_step=False)
            loss_rmse += loss_ #
        self.log(f'val_mse', loss_mse, batch_size=output.shape[0], on_epoch=True)
        self.log(f'val_rmse', loss_rmse, batch_size=output.shape[0], on_epoch=True)


        # import pdb;pdb.set_trace()
        # for name, metric in self.metrics.items(): 
        #     loss = 0
        #     if name != 'mse':
        #         metric = metric.to(gt.device)
        #     for month in range(self.horizon):
        #         if name == 'explained_var':
        #             loss_ = metric(output[:, :, month].transpose(1,0), gt[:, :, month].transpose(1, 0))
        #         else:
        #             loss_ = metric(output[:, :, month], gt[:, :, month]) 
        #         self.log(f'val_{name}_month{month}', loss_, batch_size=output.shape[0],  on_epoch=True, on_step=False)
        #         loss += loss_ # sum the losses for each month
        #     if name == 'explained_var':
        #         loss = loss/self.horizon
        #     self.log(f'val_{name}', loss, batch_size=output.shape[0], on_epoch=True)
        
            


    def log_edge_attr_distribution(self, graph_batch):
        """
        Log the distribution of edge attributes to wandb.
        """
        self.trainer.logger.experiment.log({'m2m_max_abs_edge_attr (last forecasting step)': 
                wandb.Histogram(graph_batch.edge_attr.numpy()), 
                                'epoch': self.trainer.current_epoch,
                                                    'global_step': self.global_step})
        # self.trainer.logger.experiment.log({'m2g_max_abs_edge_attr (last forecasting step)': 
        #         wandb.Histogram(graph_batch['mesh', 'to', 'grid'].edge_attr.numpy()), 
        #         'epoch': self.trainer.current_epoch,
        #         'global_step': self.global_step}) 
        # self.trainer.logger.experiment.log({'g2m_max_abs_edge_attr (last forecasting step)': 
        #         wandb.Histogram(graph_batch['grid', 'to', 'mesh'].edge_attr.numpy()), 
        #         'epoch': self.trainer.current_epoch,
        #                                                 'global_step': self.global_step})
        
 
    def on_validation_end(self):
        '''Plotting intermediate TWSA predictions and graphs to wandb.
        TWSA plots show the first prediction in the last batch of the validation set.
        Corresponding mesh2mesh adjacency matrix is plotted for each month.'''
        self.attention = self.model.attention_weights if self.model.attention_weights else None
        if self.plot_intermediate:
            if (self.trainer.current_epoch % self.plot_intermediate)==0:
                                
                predictions = self.last_validation_step_output.reshape(-1, self.num_nodes, self.horizon)
                targets = self.last_validation_step_target.reshape(-1, self.num_nodes, self.horizon)
                pred = self._twsa_unnormalizer(predictions[0]) # get the prediction of the first batch shape [batch_size,num nodes, horizon]
                true = self._twsa_unnormalizer(targets[0]) # get the ground truth of the first batch shape [batch_size,num nodes, horizon]

              
                for i in range(self.horizon):
                    twsa_pred = pred[:,i].float()
                    twsa_true = true[:,i].squeeze() 
                    self.plot_predictions(twsa_pred, twsa_true, month=i)
                    
                    graph = self.last_validation_step_graph[i]
                    # self.plot_intermediate_adjacency_from_edges(graph, num_m2m_edges, month=i)
                    if self.attention is not None:
                          # plotting predictions and corresponding mesh to mesh graph
                        batch_size = predictions.shape[0]
                        num_m2m_edges = int(self.last_validation_step_graph[0].edge_index.shape[1]/batch_size)
                
                        self.plot_attention(graph, num_m2m_edges, month=i)

    
    def on_test_start(self):
        self.num_nodes = self.trainer.datamodule.data_test.num_land_nodes()
        self.ocean_clusters = self.trainer.datamodule.num_ocean_clusters
        self.antarctica_mask = self._get_relevant_land().to(next(self.parameters()).device).bool() # shape [num nodes]
       
    def test_step(self, batch, batch_idx):
        output, _, _ = self(batch)        
        batch_size = batch.batch_size
        # antarctica_mask = self.antarctica_mask.to(output.device).bool()

        loss=0
        output = output.reshape(batch_size, self.num_nodes, self.horizon)[:, self.antarctica_mask] # reshape to [batch_size, num nodes, horizon]
        gt = batch['grid'].y.reshape(batch_size, self.num_nodes, self.horizon)[:, self.antarctica_mask] # reshape to [batch_size, num nodes, horizon] # -20 to ignore the sea pixels

        loss_mse=0
        loss_rmse=0
        for month in range(self.horizon):
            loss_ = self.metrics['mse'](output[:, :, month], gt[:, :, month]) 
            self.log(f'val_mse_month{month}', loss_, batch_size=output.shape[0],  on_epoch=True, on_step=False)
            loss_mse += loss_ # sum the losses for each month
            _rmse = self.metrics['rmse'].to(gt.device)
            loss_ = _rmse(output[:, :, month], gt[:, :, month])
            self.log(f'val_rmse_month{month}', loss_, batch_size=output.shape[0],  on_epoch=True, on_step=False)
            loss_rmse += loss_ #
        self.log(f'val_mse', loss_mse, batch_size=output.shape[0], on_epoch=True)
        self.log(f'val_rmse', loss_rmse, batch_size=output.shape[0], on_epoch=True)


        
        # for name, metric in self.metrics.items(): 
        #     loss = 0
        #     if name != 'mse':
        #         metric = metric.to(gt.device)
        #     for month in range(self.horizon):
        #         if name == 'explained_var':
        #             import pdb;pdb.set_trace()
        #             loss_ = metric(output[:, :, month].transpose(1,0), gt[:, :, month].transpose(1, 0))
        #         else:
        #             loss_ = metric(output[:, :, month], gt[:, :, month]) 
        #         self.log(f'test_{name}_month{month}', loss_, batch_size=output.shape[0],  on_epoch=True, on_step=False)
        #         loss += loss_ # sum the losses for each month
        #     if name == 'explained_var':
        #         loss = loss/self.horizon
        #     self.log(f'test_{name}', loss, batch_size=output.shape[0], on_epoch=True)


    
    
    def predict_step(self, batch, batch_idx):
        """
        This method is called when using the predict method of the trainer.
        It returns the predictions with shape [batch_size*nodes, horizon].
        """
        output, heterographs, gnn_outputs = self(batch)
        self.prediction_gnn_outputs = gnn_outputs # save the last gnn outputs for visualization
        output = self._twsa_unnormalizer(output)
        y = self._twsa_unnormalizer(batch['grid'].y)
        self.attention = self.model.attention_weights if hasattr(self.model, 'attention') else None
        return output, y, batch, heterographs, gnn_outputs # Return the prediction, true value and the last input
        
    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return torch.optim.AdamW(self.parameters(), 
                                 lr=self.learning_rate, weight_decay=1e-2)
    
    def _get_lat_weight(self):
         # compute latitude weights for the land pixels (apply antarctica mask to match later indexing)
        lat_rad = self.lat * np.pi / 180.0
        lat_w = torch.cos(lat_rad).to(next(self.parameters()).device).float()
        # keep only the land nodes used later (after masking out Antarctica)
        self.lat_weights = lat_w[self.antarctica_mask].clone()
        # normalize so the weights sum to 1 (avoid changing overall loss scale)
        self.lat_weights = self.lat_weights / (self.lat_weights.sum() + 1e-12)
        return self.lat_weights
    
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
    
    def _twsa_unnormalizer(self, data):
        if self.trainer is not None and self.trainer.datamodule is not None:
            return self.trainer.datamodule.unnormalize_twsa(data)
        else: 
            Warning('No DataModule found.')
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
            Warning('No DataModule found.')
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
        
        vmax = 40
        f, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(5,10))
        p1 = ax1.scatter(lon, lat, c=predictions, cmap='coolwarm', s=1,vmin=-vmax, vmax=vmax)
        p2 = ax2.scatter(lon, lat, c=targets, cmap='coolwarm', s=1,vmin=-vmax, vmax=vmax)
        ape = (predictions-targets)/(targets + 1e-8) * 100  # Avoid division by zero
        p3 = ax3.scatter(lon, lat, c=ape, cmap='RdBu',s=1, vmax=100, vmin=-100)
        #p3 = ax3.imshow(abs(y_hat-y_true)/(y_true+ 1e-5), cmap='Reds', vmin=0, vmax=1)
        
        ax1.set_title('Predicted TWSA')
        ax2.set_title('True TWSA')
        ax3.set_title('Percentage Difference')
        epoch = self.trainer.current_epoch
        f.suptitle(f'TWSA month={month} (Val Set)')
        
        plt.tight_layout()
        plt.colorbar(p1, ax=ax1)
        plt.colorbar(p2, ax=ax2)
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

    def plot_attention(self, graph, num_m2m_edges, month):      
        edges = graph.edge_index[:,:num_m2m_edges].cpu()
        # attentions in list are of shape [num_edges*batchsize, num attention heads], so we take the mean
        # list of attention_weights per timesteps 
        att = self.model.attention_weights[month][:num_m2m_edges].mean(dim=1).cpu() #mean over all heads
        if self.trainer is not None:  
            adj = to_dense_adj(edges, edge_attr=att)[0]
            plt.imshow(adj, cmap='magma_r', interpolation='none')
            # plt.imshow(adj, cmap='RdBu',norm=matplotlib.colors.TwoSlopeNorm(0), interpolation='none')
            plt.colorbar()
            plt.title(f'Mean Attention Month {month+1}')
        
            self.trainer.logger.experiment.log({f'Attention month {month+1}': wandb.Image(plt),
                                        'epoch': self.trainer.current_epoch,
                                        'global_step': self.global_step})
            plt.close()
        

    
    
    
    def model_summary(self):
        """
        Print the model summary.
        """
        from torchinfo import summary
        print(summary(self.model, verbose=0, depth=2))
   
# Example usage
# model = ConvLSTMLightningModule(input_dim=64, hidden_dim=[16, 32], kernel_size=(3, 3), num_layers=2)

# Example usage
# Create a PyTorch Lightning Trainer and fit your model
# trainer = pl.Trainer(max_epochs=10)
# model = ConvLSTMLightningModule(input_dim=3, hidden_dim=[64, 64], kernel_size=(3, 3), num_layers=2)
# trainer.fit(model, train_dataloder, val_dataloder)