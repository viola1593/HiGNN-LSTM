import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics
import tsl
# from tsl.metrics.torch.metrics import MaskedMAPE
from torchmetrics.regression import MeanAbsolutePercentageError
from losses.kge import KGE
from losses.utils_for_nans import mean_with_nans, mean_abs_with_nans
from tsl.ops.connectivity import adj_to_edge_index
from torch_geometric.utils import to_dense_adj




from tsl.nn.models.stgn.rnn_gcn_model import RNNEncGCNDecModel

class RNNEncDecModule(pl.LightningModule):
    def __init__(self, num_input_features,
               
                 num_hidden_features,
                 output_dim,
                 output_seq_len,  
                 node_embedding_size,
                 gcn_layers,
                 rnn_layers=1,
                 exog_size=0,
                        learning_rate=1e-3, 
                        data_image_shape=(36, 72),
                        land_sea_mask=None,
                        learn_adj=False,
            ):
        
        """
         Initializes the .
                Args:
                    ""num_input_features (int): Number of input features.
                    num_hidden_features (int): Number of hidden units in each hidden layer.
                    output_dim (int): Dimension of the output.
                    output_seq_len (int): Length of the output sequence.
                    enc_layers (int): Number of layers in the GRU encoder.
                    gcn_layers (int): Number of GCN layers in GCN decoder.
                    norm (str, optional): Normalization used by the graph convolutional layers. Default is 'mean'.
                    encode_edges (bool, optional): Whether to encode edges. Default is False.
                    exog_size (int, optional): Size of exogenous features. Default is 0.
                    activation (str, optional): Activation function. Default is 'softplus'.
                    learning_rate (float, optional): Learning rate. Default is 1e-3.
                    data_image_shape (tuple, optional): Shape of the data image. Default is (36, 72) (global map in 5° resolution.).
                    land_sea_mask (np.array, optional): Land-sea mask. Default is None.
                    learn_adj: bool, if True the model will learn the adjacency matrix, otherwise it will use a predefined adjacency matrix from the datamodule
                    static_feat: np.array of shape (batch, num_nodes) as embeddinf for the edgestructure learner
                    max_num_edges (int, optional): Maximum number of edges in the graph. Default is 10.
                    graph_edge_encoding_dim (int, optional): Dimension of the graph edge encoding. Default is 16.
                    alpha1 (float, optional): Alpha1 parameter for the edge structure learner. Default is 0.1.
                    alpha2 (float, optional): Alpha2 parameter for the edge structure learner. Default is 2.0.
        """""
        super(RNNEncDecModule, self).__init__()
        self.output_seq_len = output_seq_len
        # data specific parameters
        self.data_image_shape = data_image_shape
        self.lsm = land_sea_mask
        if self.lsm is not None:
            self.lsm = torch.Tensor(self.lsm)
        
        
        self.model = RNNEncGCNDecModel(input_size=num_input_features,
                                    hidden_size=num_hidden_features,
                                    output_size=output_dim,
                                    horizon=output_seq_len,
                                    exog_size=exog_size,
                                    rnn_layers=rnn_layers,
                                    gcn_layers=gcn_layers,
                                    rnn_dropout=0.2,
                                    gcn_dropout=0.2,
                                    cell_type='gru',
                                    activation='elu'
                                    )

        self.learning_rate = learning_rate
        self.criterion = nn.HuberLoss()
        self.metrics = {'mape' : MeanAbsolutePercentageError(), 
                        'rmse' : nn.MSELoss(), 
                        # 'ssim' : torchmetrics.image.ssim.StructuralSimilarityIndexMeasure()
                        }
        
        self.adj = None
        
        self.validation_step_outputs = []
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        x: [batches steps nodes features]
        """
        # x: [batches steps nodes features]
         
        # pred: [batches steps nodes features=1] --> [batches steps nodes]
        pred = self.model(x, edge_index, edge_weight).squeeze(dim=-1)
        
        return pred  
    
    def training_step(self, batch, batch_idx):
        
        x, y = torch.Tensor(batch['x_sequence']), torch.Tensor(batch['y_timestep'])
        twsa_last = x[:,-1,:, 0].unsqueeze(1) # get last timestep of twsa (twsa should be at index 0)
        edge_index = torch.Tensor(batch['edge_index'][0]).long()
        edge_weight = torch.ones(edge_index.shape[1]).to(x.device)
        output = self(x,edge_index, edge_weight)
        
        #output = pred_residue + twsa_last
        # output = output.squeeze()
        # output shape: [batch, seq_len, num_nodes, num_features=1]
        # import pdb; pdb.set_trace()
        
        loss = self.criterion(output[:,:,:-20], y[:,:,:-20]) # -20 to ignore the sea pixels
        self.log('train_loss', loss)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        
        x, y = torch.Tensor(batch['x_sequence']), torch.Tensor(batch['y_timestep'])
        twsa_last = x[:,-1,:, 0].unsqueeze(1) # get last timestep of twsa (twsa should be at index 0)
        edge_index = torch.Tensor(batch['edge_index'][0]).long()
        edge_weight = torch.ones(edge_index.shape[1]).to(x.device)
        output = self(x,edge_index, edge_weight)
        # import pdb; pdb.set_trace()
        #output = pred_residue + twsa_last
        output = output.squeeze()
        y = y.squeeze()
        
        for metric_name, metric in self.metrics.items():
            metric = metric.to(output.device)
            if metric_name == 'ssim':
                loss = metric(output, y) 
            else:
                loss = metric(output, y)
            if metric_name == 'rmse':
                loss = torch.sqrt(loss)
            
            self.log(f'val_{metric_name}', loss)
        # import pdb; pdb.set_trace()
        self.validation_step_outputs.append((output[:,:-20], y[:,:-20])) # last 20 pixels are sea pixels
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
        twsa_mean, twsa_std = self.trainer.datamodule.scaling_mean_std[0]
        predictions = predictions*torch.Tensor(twsa_std.values) + torch.Tensor(twsa_mean.values)
        targets = targets*torch.Tensor(twsa_std.values) + torch.Tensor(twsa_mean.values)

        kge = KGE(predictions, targets, self.lsm) # KGE only makes sense for land pixels -> mask out sea pixels otherwise the mean will always be 0 or inf
        self.logger.experiment.log({'mean_val_kge': mean_with_nans(kge[0])}) # normal logging throws an error
        self.logger.experiment.log({'mean_val_corr': mean_abs_with_nans(kge[1])}) # rmse with it self since correlation can also be negative
        
        self.logger.experiment.log({'mean_val_alpha': mean_abs_with_nans(kge[2])})
        self.logger.experiment.log({'mean_val_beta': mean_abs_with_nans(kge[3])})



        
        
    def test_step(self, batch, batch_idx):
        x, y = torch.Tensor(batch['x_sequence']), torch.Tensor(batch['y_timestep'])
        twsa_last = x[:,-1,:, 0].unsqueeze(1)
        edge_index = torch.Tensor(batch['edge_index'][0]).long()
        edge_weight = torch.ones(edge_index.shape[1]).to(x.device)
        output = self(x,edge_index, edge_weight)
        #output = pred_residue + twsa_last
        output = output.squeeze()
        y = y.squeeze()

        for metric_name, metric in self.metrics.items():
            metric = metric.to(output.device)
            if metric_name == 'ssim':
                loss = metric(output, y)
            else:
                loss = metric(output, y)
            if metric_name == 'rmse':
                loss = torch.sqrt(loss)
            
            self.log(f'val_{metric_name}', loss)
        # loss = self.criterion(output, y)
        # self.log('test_loss', loss)
    
    def predict_step(self, batch, batch_idx):
        
        x, y = torch.Tensor(batch['x_sequence']), torch.Tensor(batch['y_timestep'])
        twsa_last = x[:,-1,:, 0].unsqueeze(1)
        edge_index = torch.Tensor(batch['edge_index'][0]).long()
        edge_weight = torch.ones(edge_index.shape[1]).to(x.device)
        output = self(x,edge_index, edge_weight)
        #output = pred_res + twsa_last

        #output = output.squeeze()
        
        return output, y, x # Return the prediction, true value and the last input
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# Example usage
# model = ConvLSTMLightningModule(input_dim=64, hidden_dim=[16, 32], kernel_size=(3, 3), num_layers=2)

# Example usage
# Create a PyTorch Lightning Trainer and fit your model
# trainer = pl.Trainer(max_epochs=10)
# model = ConvLSTMLightningModule(input_dim=3, hidden_dim=[64, 64], kernel_size=(3, 3), num_layers=2)
# trainer.fit(model, train_dataloder, val_dataloder)