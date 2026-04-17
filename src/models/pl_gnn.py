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
from torch_geometric.utils import to_dense_adj



from models.stgcn import STConv

class GraphSTConvModule(pl.LightningModule):
    def __init__(self, num_nodes: int, in_channels: int, input_seq_len: int, hidden_channels: int, out_channels: int,
                        output_seq_len: int, K: int, normalization: str = "sym", bias: bool = True, 
                        learning_rate=1e-3):
        
        super(GraphSTConvModule, self).__init__()
        self.output_seq_len = output_seq_len
        self.input_seq_len = input_seq_len
        kernel_size = (self.input_seq_len-self.output_seq_len)//2+1 # This is given by the implementation of STConv
        self.model = STConv(num_nodes=num_nodes,
                                in_channels=in_channels,
                                hidden_channels=hidden_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                K=K,
                                normalization=normalization,
                                bias=bias)

        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        
    def forward(self, x, edge_index):
        """Forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph.

        Arg types:
            * **x** (PyTorch FloatTensor) - Sequence of node features of shape (Batch size X Input time steps X Num nodes X In channels).
            * **edge_index** (PyTorch LongTensor) - Graph edge indices.
            * **edge_weight** (PyTorch LongTensor, optional)- Edge weight vector.

        Return types:
            * **T** (PyTorch FloatTensor) - Sequence of node features of shape (Batch size X Input time steps X Num nodes X out channels).
        """

        pred = self.model(x, edge_index)
        return pred  
    
    def training_step(self, batch, batch_idx):
        
        x, y = torch.Tensor(batch['x_sequence']), torch.Tensor(batch['y_timestep'])
        edge_index = torch.Tensor(batch['edge_index'][0]).long()
        output = self(x,edge_index)
        output = output.squeeze()
        loss = self.criterion(output[:,-1,:], y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # import pdb; pdb.set_trace()
        x, y = torch.Tensor(batch['x_sequence']), torch.Tensor(batch['y_timestep'])
        edge_index = torch.Tensor(batch['edge_index'][0]).long()
        output = self(x,edge_index)
        output = output.squeeze()
        loss = self.criterion(output[:,-1,:], y)
        self.log('val_loss', loss)
        
    def test_step(self, batch, batch_idx):
        x, y = torch.Tensor(batch['x_sequence']), torch.Tensor(batch['y_timestep'])
        edge_index = torch.Tensor(batch['edge_index'][0]).long()
        output = self(x,edge_index)
        output = output.squeeze()
        loss = self.criterion(output[:,-1,:], y)
        self.log('test_loss', loss)
    
    def predict_step(self, batch, batch_idx):
        
        x, y = torch.Tensor(batch['x_sequence']), torch.Tensor(batch['y_timestep'])
        edge_index = torch.Tensor(batch['edge_index'][0]).long()
        output = self(x,edge_index)
        output = output.squeeze()
        
        return output[-1,:], y.squeeze(), x # Return the prediction, true value and the last input
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)



from tsl.nn.models.stgn.gru_gcn_model import GRUGCNModel
from models.layers.graphino_structure import EdgeStructureLearner

class GRUGCModule(pl.LightningModule):
    def __init__(self, num_input_features,
               
                 num_hidden_features,
                 output_dim,
                 output_seq_len,  
                 enc_layers,
                 gcn_layers,
                 norm='mean',
                 encode_edges=False,
                 exog_size=0,
                 activation='softplus', 
                        learning_rate=1e-3, 
                        data_image_shape=(36, 72),
                        land_sea_mask=None,
                        learn_adj=True,
                        graph_construction='KNN', 
                        static_feat=None, 
                        max_num_edges=10, 
                        graph_edge_encoding_dim=16,
                        alpha1=0.1, alpha2=2.0):
        
        """
         Initializes the GRUGCModule.
                Args:
                    num_input_features (int): Number of input features.
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
        """
        super(GRUGCModule, self).__init__()
        self.output_seq_len = output_seq_len
        # data specific parameters
        self.data_image_shape = data_image_shape
        self.lsm = land_sea_mask
        if self.lsm is not None:
            self.lsm = torch.Tensor(self.lsm)
        

        self.learn_adj = learn_adj
        if self.learn_adj:
            self.graph_learner = EdgeStructureLearner(num_nodes=24737, #data_image_shape[0]*data_image_shape[1], 
                                                  max_num_edges=max_num_edges, 
                                                  dim=graph_edge_encoding_dim, # This is the dimension of the node embeddings 
                                                  static_feat=static_feat, 
                                                 alpha1=alpha1, alpha2=alpha2,
                                                  self_loops=True)
                
        
        
        self.model = GRUGCNModel(input_size=num_input_features,
                                    hidden_size=num_hidden_features,
                                    output_size=output_dim,
                                    horizon=output_seq_len,
                                    exog_size=exog_size,
                                    enc_layers=enc_layers,
                                    gcn_layers=gcn_layers,
                                    norm=norm,
                                    encode_edges=encode_edges,
                                    activation=activation)

        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        self.metrics = {'mape' : MeanAbsolutePercentageError(), 
                        'rmse' : nn.MSELoss(), 
                        #'ssim' : torchmetrics.image.ssim.StructuralSimilarityIndexMeasure()
                        }
        
        self.validation_step_outputs = []
        
    def forward(self, x, edge_index):
        """
        x: [batches steps nodes features]
        """
        # x: [batches steps nodes features]
        if self.learn_adj:
            # Generate an adjacency matrix/connectivity structure for the graph convolutional forward pass
            self.adj = self.graph_learner.forward(x.device)
            edge_index, edge_weights = adj_to_edge_index(self.adj)
        else: 
            edge_weights = None
            self.adj = to_dense_adj(edge_index).squeeze()

        pred = self.model(x, edge_index, edge_weight=edge_weights)
        return pred  
    
    def training_step(self, batch, batch_idx):
        
        x, y = torch.Tensor(batch['x_sequence']), torch.Tensor(batch['y_timestep'])
        edge_index = torch.Tensor(batch['edge_index'][0]).long()
        output = self(x,edge_index)
        output = output.squeeze()
        loss = self.criterion(output, y.squeeze())
        self.log('train_loss', loss)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x, y = torch.Tensor(batch['x_sequence']), torch.Tensor(batch['y_timestep'])
        edge_index = torch.Tensor(batch['edge_index'][0]).long()
        output = self(x,edge_index)
        output = output.squeeze()
        
        for metric_name, metric in self.metrics.items():
            metric = metric.to(output.device)
            if metric_name == 'ssim':
                loss = metric(output.reshape(output.shape[0], self.data_image_shape[0], self.data_image_shape[1]).unsqueeze(1), 
                              y.reshape(output.shape[0], self.data_image_shape[0], self.data_image_shape[1]).unsqueeze(1)) # 36 72 is the shape of the image
            else:
                loss = metric(output, y.squeeze())
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
        edge_index = torch.Tensor(batch['edge_index'][0]).long()
        output = self(x,edge_index)
        output = output.squeeze()

        for metric_name, metric in self.metrics.items():
            metric = metric.to(output.device)
            if metric_name == 'ssim':
                loss = metric(output.reshape(output.shape[0], self.data_image_shape[0], self.data_image_shape[1]).unsqueeze(1), 
                              y.reshape(output.shape[0], self.data_image_shape[0], self.data_image_shape[1]).unsqueeze(1))
            else:
                loss = metric(output, y.squeeze())
            if metric_name == 'rmse':
                loss = torch.sqrt(loss)
            
            self.log(f'val_{metric_name}', loss)
        # loss = self.criterion(output, y)
        # self.log('test_loss', loss)
    
    def predict_step(self, batch, batch_idx):
        
        x, y = torch.Tensor(batch['x_sequence']), torch.Tensor(batch['y_timestep'])
        edge_index = torch.Tensor(batch['edge_index'][0]).long()
        output = self(x,edge_index)
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