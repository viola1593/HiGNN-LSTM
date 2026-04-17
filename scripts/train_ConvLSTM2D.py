
import argparse
import os
from datetime import datetime
import torch
from typing import Any, Dict
import yaml
import json
from models.pl_convlstm import ConvLSTMModule
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import pytorch_lightning as pl
import torchmetrics
import wandb
# from lightning.pytorch.callbacks import EarlyStopping

from datamodules.TWSA_datamodule2D import TWSA_Datamodule2D_new 
from losses.kge import KGE



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
    loggers = [pl.loggers.WandbLogger(
                    name=config["experiment"]["save_dir"].split("/")[-1],
                    save_dir=config["experiment"]["save_dir"],
                    project=config["wandb"]["project"],
                    # entity=config["wandb"]["entity"],
                    resume="allow",
                    config=config,
                    mode=config["wandb"].get("mode", "online"))
                ]
    
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=config["experiment"]["save_dir"], 
                                                       save_top_k=1, monitor="val_rmse")

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

def plot_prediction(y_hat, y_true, y_last, datamodule, name='0', save_dir=None, trainer=None, title=None):
    
    # shape = (len(datamodule.data_pred.ds_local.lat),len(datamodule.data_pred.ds_local.lon))
    # # import pdb; pdb.set_trace()
    # mean, std = datamodule.scaling_mean_std[0][0].values, datamodule.scaling_mean_std[0][1].values
    mask = datamodule.data_pred.land_ocean_mask[0] & datamodule.data_pred.mask_not_antarctica[0] # [SeqLen, H, W]
    # y_hat = (y_hat.reshape(shape)*std + mean)*mask
    # y_true = (y_true.reshape(shape)*std + mean)*mask
    # y_last = (y_last.reshape(shape)*std + mean)*mask
    f, (ax0, ax1, ax2, ax3) = plt.subplots(4,1,figsize=(5,10))
    vmax=40
    p0 = ax0.imshow(y_last, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    p1 = ax1.imshow(y_hat, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    p2 = ax2.imshow(y_true, cmap='coolwarm', vmin=-vmax, vmax=vmax )
    ape = (abs((y_hat-y_true)/(y_true+ 1e-10)))*mask *100
    p3 = ax3.imshow(ape, cmap='Reds', vmax=100)
    #p3 = ax3.imshow(abs(y_hat-y_true)/(y_true+ 1e-5), cmap='Reds', vmin=0, vmax=1)
    ax0.set_title('Last TWSA')
    ax1.set_title('Predicted TWSA')
    ax2.set_title('True TWSA')
    ax3.set_title('Percentage Difference')
    f.suptitle(title)
    
    plt.tight_layout()
    plt.colorbar(p0, ax=ax0)
    plt.colorbar(p1, ax=ax1)
    plt.colorbar(p2, ax=ax2)
    plt.colorbar(p3, ax=ax3)

    if save_dir is not None:
        plt.savefig(save_dir)
        plt.close()
    elif trainer is not None:
        trainer.logger.experiment.log({name: wandb.Image(plt)})
        plt.close() 


def plot_KGE(kge, datamodule, save_dir=None):
    
    f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(20,10))
    # -0.41 everything below is worse than a mean prediction for each time step
    p0 = ax0.imshow(kge[0],  vmin=-0.41)
    plt.colorbar(p0, ax=ax0, extend='min')
    p1 = ax1.imshow(kge[1], vmin=-1, vmax=1, cmap='coolwarm')
    plt.colorbar(p1, ax=ax1)
    p2 = ax2.imshow(kge[2],   norm=matplotlib.colors.LogNorm( vmax=10,))

    plt.colorbar(p2, ax=ax2, extend='max')
    p3 = ax3.imshow(kge[3],  cmap='coolwarm', vmin=0, vmax=2)
 
    plt.colorbar(p3, ax=ax3, extend='both')
    ax0.set_title('KGE')
    ax1.set_title('Correlation')
    ax2.set_title('Alpha ($Std_{pred}/Std_{true}$)')
    ax3.set_title('Beta ($Mean_{pred}/Mean_{true}$)')

    f.suptitle('Pixelwise Means over Validation sequences')
    # add text to the plot in lower right corner
    plt.text(0.5, -0.1, 'Min/Max Alpha: {:.2f} / {:.2f}'.format(np.nanmin(kge[2]), np.nanmax(kge[2])),  horizontalalignment='center',

     verticalalignment='center', transform=ax3.transAxes)
    plt.text(0.5, -0.2, 'Min/Max Beta: {:.2f} / {:.2f}'.format(np.nanmin(kge[3]), np.nanmax(kge[3])),  horizontalalignment='center',

     verticalalignment='center', transform=ax3.transAxes)
    
    plt.tight_layout()

    plt.savefig(save_dir)
    plt.close()


def plot_explained_variance(y_hat, y_true, datamodule, trainer=None):
    seq_len = datamodule.target_seq_len
    lon, lat = y_true.shape[1], y_true.shape[2]
    
    # flip the cmap colors and add color for below 0 values
    cmap = plt.cm.Reds.reversed()
    cmap.set_under('blue')
    exp_var = torchmetrics.ExplainedVariance(multioutput='raw_values')(y_hat, y_true)
#    exp_var = exp_var.reshape(lon, lat)
    f, ax = plt.subplots()
    p = ax.imshow(exp_var, cmap=cmap, vmin=0)
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

    twsa_sequence_data = TWSA_Datamodule2D_new(**config['datamodule'])
    
    
        #load model from checkpoint
    if config["load_checkpoint"]:
        twsa_sequence_data.setup(stage='fit')
        model = ConvLSTMModule.load_from_checkpoint(config["load_checkpoint"], **config["model"])
    else:
        # generate model
        model = ConvLSTMModule(**config["model"], kernel_size=(3,3),                             
                                # sequence_length=twsa_sequence_data.input_seq_len, 
                                horizon=twsa_sequence_data.target_seq_len)
    

        trainer.loggers[0].watch(model, log='all')
        trainer.log_every_n_steps = config["pl"]["log_every_n_steps"] # train dataset is only defined during training, after setup is done, so len of dataloader is unknown for now
        
        trainer.fit(model, datamodule=twsa_sequence_data)

    # save config
    save_config(config, os.path.join(config["experiment"]["save_dir"], "config.yaml"))
    
    # --------------------------------------------------------------------------------------------------------------
    # Calculate metrics on validation set
    # --------------------------------------------------------------------------------------------------------------
    # 
    metrics_val = trainer.validate(model, datamodule=twsa_sequence_data)
    metrics_dict={'validation_metrics': metrics_val}
    
    
    
 # --------------------------------------------------------------------------------------------------------------
 # Plotting predictions
 # --------------------------------------------------------------------------------------------------------------
 #  
    if not config['debug']:
        if config['datamodule']['output_seq_len'] > 1:
            # plot prediction of last sequence in validation time series
            
            if config['pl']['overfit_batches']>0:
                preds, true, input = trainer.predict(model, dataloaders=twsa_sequence_data.predict_dataloader(time_step_idx=0, period=twsa_sequence_data.train_val_test_periods['train']))[0]
                title = 'TWSA Prediction on Train Set'
            else:
                preds, true, input = trainer.predict(model, 
                                                     dataloaders=twsa_sequence_data.predict_dataloader(time_step_idx=100, 
                                                                                                   period=twsa_sequence_data.train_val_test_periods['val']))[0]
                title = 'TWSA Prediction on Val Set'   
            for i in range(config['datamodule']['output_seq_len']):
                #import pdb; pdb.set_trace()
                # (Batch size(=1) X time steps X H X W)
                
                twsa_pred = preds[0, i].float()
                twsa_true = true[0, i]
                if i ==0:
                    last_twsa = input[0, -1, 0] # get last time step of input data
                    last_twsa = twsa_sequence_data.unnormalize_twsa(last_twsa)
                else:
                    last_twsa = preds[0, i-1].float()
                plot_prediction(twsa_pred, twsa_true, last_twsa, twsa_sequence_data, save_dir=None, trainer=trainer, name=f'Prediction_month_{i}', title=title)
                # plot explained variance
            plot_explained_variance(preds[0], true[0], twsa_sequence_data, trainer=trainer)
            pixel_kge = KGE(preds[0], true[0]).cpu() # transpose to have seq_len first
            plot_KGE(pixel_kge.reshape(4, 180, 360), twsa_sequence_data, save_dir=os.path.join(config["experiment"]["save_dir"], f'kge_for_preds.png'))
            


        else:  # plotting single time step predictions
            for i in [12, 48, 64, 100]:
                
                preds, true, input, graphs = trainer.predict(model, dataloaders=twsa_sequence_data.predict_dataloader(time_step_idx=i, period=twsa_sequence_data.train_val_test_periods['val']))[0]
                
                full_twsa_pred = preds.float()
                full_twsa_true = true 
                # (Batch size X Input time steps X Num nodes X In channels) --> (Batch size X Input time steps X In channels X Num nodes) 
                last_twsa = torch.Tensor(input).permute(0,1,3,2)[:,-1,0] # (batch_size(=1), seq_len, n_features) TWSA at position 0

                plot_prediction(full_twsa_pred,full_twsa_true, last_twsa, twsa_sequence_data, save_dir=os.path.join(config["experiment"]["save_dir"], f'global_lstm_on_val{i}.png'))
            

        pixel_kge_mean=np.nanmean(pixel_kge.float(), axis=1).astype('float')

        pixel_kge_dict = {
            'kge': pixel_kge_mean[0],
            'correlation': pixel_kge_mean[1],
            'alpha': pixel_kge_mean[2],
            'beta': pixel_kge_mean[3]
        }
        # import pdb; pdb.set_trace()
        metrics_dict['val_kge']= pixel_kge_dict
    


    with open(os.path.join(config["experiment"]["save_dir"], "eval_on_test.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)

        



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
