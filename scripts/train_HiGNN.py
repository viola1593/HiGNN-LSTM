import argparse
from datetime import datetime
import json
import matplotlib.pyplot as plt
import os
import torch
from typing import Any, Dict
import yaml
import wandb

import pytorch_lightning as pl

from datamodules.TWSA_heterograph_datamodule import TWSA_HeteroGraphDatamodule
from models.pl_hmmodel import HMModule
from utils.utils import read_config, save_config





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
    

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=config["experiment"]["save_dir"], 
                                                       save_top_k=1, monitor="val_rmse",
                                                       save_last=True)
    callbacks = [checkpoint_callback]
    if not config["debug"]: # LearningRateMonitor needs a logger
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="epoch"))

    return pl.Trainer(
        **config["pl"],
        default_root_dir=config["experiment"]["save_dir"],
        logger=loggers,
        callbacks=callbacks
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
        f"_{config['datamodule']['graph_variant']}"
        f"_seed{config['experiment']['seed']}"
        f"_{datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}"
    )
    exp_dir_path = os.path.join(config["experiment"]["exp_dir"], exp_dir_name)
    os.makedirs(exp_dir_path)
    config["experiment"]["save_dir"] = exp_dir_path

    return config


def plot_prediction(y_hat, y_true, y_last, 
                    datamodule, save_dir=None,
                      trainer=None, 
                      name='0', 
                      title='TWSA Prediction on Train Set',
                      antarctica_mask=None):
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


def train(config_path, seed=42):
    """Training Script."""
    # torch.set_float32_matmul_precision("medium")

    config = read_config(config_path)
    config["experiment"]["seed"] = seed
    pl.seed_everything(seed, workers=True)
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
                              use_ocean_nodes=twsa_sequence_data.use_ocean_nodes,
                              sequence_length=twsa_sequence_data.input_seq_len, 
                              output_seq_len=twsa_sequence_data.target_seq_len, 
                              land_sea_mask=None)
        if not config["debug"]:
            trainer.loggers[0].watch(model, log='all')
        trainer.fit(model, datamodule=twsa_sequence_data, ckpt_path=(config["load_checkpoint"]))
    else:
        # generate model
        
        
        model = gnn_model(**config["model"],
                              num_input_features=twsa_sequence_data.num_features,
                              num_ocean_features=twsa_sequence_data.num_ocean_features,
                              use_ocean_nodes=twsa_sequence_data.use_ocean_nodes,
                              sequence_length=twsa_sequence_data.input_seq_len, 
                              output_seq_len=twsa_sequence_data.target_seq_len, land_sea_mask=None) #static_feat=twsa_sequence_data.data_train.static_feat)#twsa_sequence_data.lsm)
        
        if not config["debug"]:
            trainer.loggers[0].watch(model, log='all')

        model.model_summary()

        trainer.fit(model, datamodule=twsa_sequence_data)
        
    # save config
    save_config(config, os.path.join(config["experiment"]["save_dir"], "config.yaml"))

    


    
 # --------------------------------------------------------------------------------------------------------------
 # Evaluate and Plotting predictions per month
 # --------------------------------------------------------------------------------------------------------------
 
    
    if not config['debug']:
        antarctica_mask = (~twsa_sequence_data.get_antarctica_landmask())
        if config['pl']['overfit_batches']>0:
            title = 'TWSA Prediction on Train Set'
            metrics_val = trainer.validate(model, dataloaders=twsa_sequence_data.train_dataloader())
            prediction_period = twsa_sequence_data.train_val_test_periods['train']
  
        else:
            title = 'TWSA Prediction on Val Set'
            metrics_val = trainer.validate(model, datamodule=twsa_sequence_data) 
            prediction_period = twsa_sequence_data.train_val_test_periods['val']
            
        with open(os.path.join(config["experiment"]["save_dir"], "metrics_eval.json"), "w") as f:
            json.dump(metrics_val, f, indent=4)

 # --------------------------------------------------------------------------------------------------------------
 # Plotting predictions per month
 # --------------------------------------------------------------------------------------------------------------
 #  
        prediction_dl = twsa_sequence_data.predict_dataloader(time_step_idx=0, 
                                                              period=prediction_period)     
        preds, true, input, graphs, _ = trainer.predict(model, 
                                                     dataloaders=prediction_dl)[0]
         
        preds = preds[antarctica_mask]
        true = true[antarctica_mask]
        
        for i in range(config['datamodule']['output_seq_len']):
            twsa_pred = preds[:,i]
            twsa_true = true[:,i]
            if i ==0: # in anomaly mode this input is twsa - C, so the first 'Last TWSA' panel shows the anomaly
                last_twsa = input['grid'].x[:,-1,0]
                last_twsa = twsa_sequence_data.unnormalize_twsa(last_twsa)[antarctica_mask]
            else:
                last_twsa = preds[:,i-1]

            
            plot_prediction(twsa_pred, twsa_true, last_twsa, twsa_sequence_data, 
                            save_dir=None, trainer=trainer, name=f'Prediction_month_{i}', title=title,
                            antarctica_mask=antarctica_mask)
         
       
        
        



def start():
    """Start training."""
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="trainer.py", description="Trains the model given config file."
    )

    parser.add_argument("--config_path", help="Path to the config file", required=True)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for weight init and shuffling")

    args = parser.parse_args()

    train(args.config_path, args.seed)


if __name__ == "__main__":
    start()
