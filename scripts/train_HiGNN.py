
import argparse
import os
from datetime import datetime
import torch
import torchmetrics
from typing import Any, Dict
import yaml
import json
from models.pl_hmmodel import HMModule

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wandb
from torch_geometric.utils import to_dense_adj

import pytorch_lightning as pl
from losses.kge import KGE
from losses.utils_for_nans import mean_with_nans, median_with_nans
# from lightning.pytorch.callbacks import EarlyStopping

from datamodules.TWSA_heterograph_datamodule import TWSA_HeteroGraphDatamodule



# Function to load yaml configuration file
def read_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config

def save_config(config, path):
    with open(path, 'w') as file:
        yaml.dump(config, file)


def generate_trainer(config: Dict[str, Any]) -> pl.Trainer:
    """Generate a pytorch lightning trainer."""
    if not config["debug"]:
        loggers = [pl.loggers.WandbLogger(
                    name=config["experiment"]["save_dir"].split("/")[-1],
                    save_dir=config["experiment"]["save_dir"],
                    project=config["wandb"]["project"],
                    # entity=config["wandb"]["entity"],
                    resume="allow",
                    config=config,
                    mode=config["wandb"].get("mode", "online"))
                    ]
    else:
        loggers = False
    
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_rmse", mode="min", 
                                                            patience=10)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=config["experiment"]["save_dir"], 
                                                       save_top_k=1, monitor="val_rmse",
                                                       save_last=True)

    return pl.Trainer(
        **config["pl"],
        default_root_dir=config["experiment"]["save_dir"],
        logger=loggers,
        callbacks=[checkpoint_callback]
        )


def create_experiment_dir(config: Dict[str, Any]) -> str:
    """Create experiment directory.

    Args:
        config: config file

    Returns:
        config with updated save_dir
    """
    os.makedirs(config["experiment"]["exp_dir"], exist_ok=True)
    exp_dir_name = (
        f"{config['experiment']['experiment_name']}"
        f"_{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}"
    )
    exp_dir_path = os.path.join(config["experiment"]["exp_dir"], exp_dir_name)
    os.makedirs(exp_dir_path)
    config["experiment"]["save_dir"] = exp_dir_path

    return config

def plot_adj_mat(adj, save_dir=None):
    plt.imshow(adj, cmap='Reds')
    plt.colorbar()
    plt.savefig(save_dir)
    plt.close()

def plot_pixel_adj_mat(adj, data_lonlat, save_dir=None):
    plt.scatter(data_lonlat[:,0], data_lonlat[:,1], c=adj, cmap='viridis', s=1)
    plt.colorbar()
    plt.savefig(save_dir)
    plt.close()

def plot_prediction(y_hat, y_true, y_last, 
                    datamodule, save_dir=None,
                      trainer=None, 
                      name='0', 
                      title='TWSA Prediction on Train Set',
                      antarctica_mask=None, 
                      overfitting=False):
    lon, lat = datamodule.data_pred.land[0][:,0], datamodule.data_pred.land[0][:,1]
    if antarctica_mask is not None:
        lon, lat = lon[antarctica_mask], lat[antarctica_mask]
    
    epsilon = 1e-8
    vmax = 100
    f, (ax0, ax1, ax2, ax3) = plt.subplots(4,1,figsize=(5,10))
    p0 = ax0.scatter(lon, lat, c=y_last, cmap='coolwarm', s=1, vmin=-vmax, vmax=vmax)
    p1 = ax1.scatter(lon, lat, c=y_hat, cmap='coolwarm', s=1,vmin=-vmax, vmax=vmax)
    p2 = ax2.scatter(lon, lat, c=y_true, cmap='coolwarm', s=1,vmin=-vmax, vmax=vmax)

    ape = torch.abs((y_true - y_hat) / (y_true + epsilon)) * 100
    p3 = ax3.scatter(lon, lat, c=ape, cmap='Reds',s=1, vmax=100)

    ax3.set_title('Absolute Percentage Difference')
    ax0.set_title('Last TWSA')
    ax1.set_title('Predicted TWSA')
    ax2.set_title('True TWSA')
    
    f.suptitle(title)
    
    plt.tight_layout()
    plt.colorbar(p0, ax=ax0, extend='both')
    plt.colorbar(p1, ax=ax1, extend='both')
    plt.colorbar(p2, ax=ax2, extend='both')
    plt.colorbar(p3, ax=ax3, extend='max')
    if save_dir is not None:
        plt.savefig(save_dir)
        plt.close()
    elif trainer is not None:
        trainer.logger.experiment.log({name: wandb.Image(plt)})
        plt.close() 

def plot_KGE(kge, datamodule, save_dir=None,
                      antarctica_mask=None):
    
    lon, lat = datamodule.data_pred.land[0][:,0], datamodule.data_pred.land[0][:,1]
    if antarctica_mask is not None:
        lon, lat = lon[antarctica_mask], lat[antarctica_mask]

   
    f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(20,10))
    # -0.41 everything below is worse than a mean prediction for each time step
    p0 = ax0.scatter(lon, lat, c=kge[0], s=1,  vmin=-0.41)
    plt.colorbar(p0, ax=ax0, extend='min')
    p1 = ax1.scatter(lon, lat,s=1, c=kge[1], vmin=-2, vmax=2, cmap='coolwarm')
    plt.colorbar(p1, ax=ax1)
    p2 = ax2.scatter(lon, lat,s=1, c=kge[2],   norm=matplotlib.colors.LogNorm( vmax=10,))

    plt.colorbar(p2, ax=ax2, extend='max')
    p3 = ax3.scatter(lon, lat, s=1, c=kge[3],  cmap='coolwarm', vmin=0, vmax=2)
 
    plt.colorbar(p3, ax=ax3, extend='both')
    ax0.set_title('KGE')
    ax1.set_title('Correlation')
    ax2.set_title('Alpha ($Std_{pred}/Std_{true}$)')
    ax3.set_title('Beta ($Mean_{pred}/Mean_{true}$)')

    f.suptitle('Pixelwise Means over Validation sequences')
    # add text to the plot in lower right corner
    plt.text(0.5, -0.1, 'Min/Max Alpha: {:.2f} / {:.2f}'.format(np.nanmin(kge[2]), np.nanmax(kge[2])),  
             horizontalalignment='center', 
             verticalalignment='center', transform=ax3.transAxes)
    plt.text(0.5, -0.2, 'Min/Max Beta: {:.2f} / {:.2f}'.format(np.nanmin(kge[3]), np.nanmax(kge[3])),  
             horizontalalignment='center',
                verticalalignment='center', transform=ax3.transAxes)
    
    plt.tight_layout()
    plt.savefig(save_dir)
    plt.close()


def plot_explained_variance(y_hat, y_true, datamodule, trainer=None,
                      antarctica_mask=None):
    lon, lat = datamodule.data_pred.land[0][:,0], datamodule.data_pred.land[0][:,1]
    if antarctica_mask is not None:
        lon, lat = lon[antarctica_mask], lat[antarctica_mask]
    # flip the cmap colors and add color for below 0 values
    cmap = plt.cm.Reds.reversed()
    cmap.set_under('blue')
    exp_var = torchmetrics.ExplainedVariance(multioutput='raw_values')(y_hat, y_true)
    f, ax = plt.subplots(1,1)
    p=ax.scatter(lon, lat, c=exp_var, cmap=cmap, s=1, vmin=0)
    ax.set_title(f'Explained Variance per Pixel')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.tight_layout()
    plt.colorbar(p, ax=ax, extend='min')
    if trainer is not None:
        trainer.logger.experiment.log({f'ExpVar': wandb.Image(plt)})
    plt.close() 



def train(config_path):
    """Training Script."""
    # torch.set_float32_matmul_precision("medium")
    
    pl.seed_everything(42, workers=True)
    config = read_config(config_path)
    config = create_experiment_dir(config)

    # generate trainer
    trainer = generate_trainer(config)

    twsa_sequence_data = TWSA_HeteroGraphDatamodule(**config['datamodule'])
    
    
    gnn_model = HMModule

    
        #load model from checkpoint
    if config["load_checkpoint"]:
        
        twsa_sequence_data.setup(stage='fit')
        model = gnn_model.load_from_checkpoint(config["load_checkpoint"], **config["model"], 
                                num_input_features=twsa_sequence_data.num_features,
                                num_ocean_features=twsa_sequence_data.num_ocean_features,
                              sequence_length=twsa_sequence_data.input_seq_len, 
                              output_seq_len=twsa_sequence_data.target_seq_len, 
                              land_sea_mask=None)
        trainer.loggers[0].watch(model, log='all')
        trainer.fit(model, datamodule=twsa_sequence_data, ckpt_path=(config["load_checkpoint"]))
    else:
        # generate model
        
        
        model = gnn_model(**config["model"],
                              num_input_features=twsa_sequence_data.num_features,
                              num_ocean_features=twsa_sequence_data.num_ocean_features,
                              sequence_length=twsa_sequence_data.input_seq_len, 
                              output_seq_len=twsa_sequence_data.target_seq_len, land_sea_mask=None) #static_feat=twsa_sequence_data.data_train.static_feat)#twsa_sequence_data.lsm)
        
        if not config["debug"]:
            trainer.loggers[0].watch(model, log='all')
            #trainer.log_every_n_steps = config["pl"]["log_every_n_steps"]
            #  # train dataset is only defined during training, after setup is done, so len of dataloader is unknown for now
        
        model.model_summary()

        trainer.fit(model, datamodule=twsa_sequence_data)
        
    # save config
    save_config(config, os.path.join(config["experiment"]["save_dir"], "config.yaml"))

 # --------------------------------------------------------------------------------------------------------------
 # Validation and Testing
 # --------------------------------------------------------------------------------------------------------------
 # 
    
    # if config['model']['gnn_method']=='simple':
    #     print('Model forecasting timestep embedding weights:')
    #     print(model.model.timestep_embeddings.weight)

    adj = to_dense_adj(twsa_sequence_data.data_train.data['mesh', 'to', 'mesh'].edge_index, 
                       edge_attr=twsa_sequence_data.data_train.data['mesh', 'to', 'mesh'].edge_attr).squeeze()
    # plotting the original adjacency matrices
    plot_adj_mat(adj, save_dir=os.path.join(config["experiment"]["save_dir"], 'original_adj_mat.png'))
    testadj= adj[0]
    
    data_lonlat = twsa_sequence_data.data_train.get_mesh_coords()   
    plot_pixel_adj_mat(testadj, data_lonlat, save_dir=os.path.join(config["experiment"]["save_dir"], 'test_adj_mat.png'))
      
    

    
    
    # kge = calculate_plot_kge_per_pixel(model.validation_step_outputs, datamodule=twsa_sequence_data, 
    #                                    save_dir=os.path.join(config["experiment"]["save_dir"], f'kge_on_val_period.png'))

 # --------------------------------------------------------------------------------------------------------------
 # Plotting predictions per month
 # --------------------------------------------------------------------------------------------------------------
 #  
    
    if not config['debug']:
        antarctica_mask = (~twsa_sequence_data.get_antarctica_landmask())
        if config['pl']['overfit_batches']>0:
            overfitting=True
            title = 'TWSA Prediction on Train Set'
            metrics_val = trainer.validate(model, dataloaders=twsa_sequence_data.train_dataloader())
            prediction_period = twsa_sequence_data.train_val_test_periods['train']
  
        else:
            overfitting=False
            title = 'TWSA Prediction on Val Set'
            metrics_val = trainer.validate(model, datamodule=twsa_sequence_data) 
            prediction_period = twsa_sequence_data.train_val_test_periods['val']
            
            
        prediction_dl = twsa_sequence_data.predict_dataloader(time_step_idx=0, 
                                                              period=prediction_period)     
        preds, true, input, graphs, _ = trainer.predict(model, 
                                                     dataloaders=prediction_dl)[0]
         
        num_ocean_clusters = twsa_sequence_data.num_ocean_clusters
        preds = preds[antarctica_mask]
        true = true[antarctica_mask]
        
        for i in range(config['datamodule']['output_seq_len']):
            twsa_pred = preds[:,i]
            twsa_true = true[:,i]
            if i ==0:
                last_twsa = input['grid'].x[:,-1,0]
                last_twsa = twsa_sequence_data.unnormalize_twsa(last_twsa)[antarctica_mask]
            else:
                last_twsa = preds[:,i-1]

            
            plot_prediction(twsa_pred, twsa_true, last_twsa, twsa_sequence_data, 
                            save_dir=None, trainer=trainer, name=f'Prediction_month_{i}', title=title,
                            antarctica_mask=antarctica_mask, 
                            overfitting=overfitting)
            if config['model']['gnn_method'] == 'gat':
                model.plot_attention(graphs[0], 
                                        graphs[0].edge_index.shape[1], 
                                        month=i)     
# --------------------------------------------------------------------------------------------------------------
 # Plotting evaluations per pixel
 # --------------------------------------------------------------------------------------------------------------
  
            
        plot_explained_variance(preds.transpose(1,0), 
                                true.transpose(1,0), 
                                twsa_sequence_data, trainer=trainer,
                                antarctica_mask=antarctica_mask)
            
        kge_per_pixel = KGE(preds.transpose(1,0), 
                            true.transpose(1,0)) # transpose to have seq_len first
        plot_KGE(kge_per_pixel, 
                twsa_sequence_data, 
                save_dir=os.path.join(config["experiment"]["save_dir"],
                                    f'kge_for_preds.png'),
                antarctica_mask=antarctica_mask)
        
        ExpVar = torchmetrics.ExplainedVariance(multioutput='raw_values')
        exp_var_per_timestep = ExpVar(preds, true)
            
           
        
        pixel_kge_mean=np.nanmean(kge_per_pixel.float(), axis=1).astype('float')

        pixel_kge_dict = {
            'kge': pixel_kge_mean[0],
            'correlation': pixel_kge_mean[1],
            'alpha': pixel_kge_mean[2],
            'beta': pixel_kge_mean[3]
        }
        
        metrics_dict={ 'validation':metrics_val, 
                      
                      'kge_plotted_seq': pixel_kge_dict, 
                      'exp_var_plotted_seq': exp_var_per_timestep.tolist()}
        
        
        with open(os.path.join(config["experiment"]["save_dir"], "metrics_eval.json"), "w") as f:
            json.dump(metrics_dict, f, indent=4)
    
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
    
    # print(model.prediction_gnn_outputs) # list of shape [batch_size*num_nodes, horizon]
    # print(model.attention)  



def start():
    """Start training."""
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="trainer.py", description="Trains the model given config file."
    )

    parser.add_argument("--config_path", help="Path to the config file", required=True)

    args = parser.parse_args()

    train(args.config_path)


if __name__ == "__main__":
    start()
