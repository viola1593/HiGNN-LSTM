
import argparse
import os
from datetime import datetime
from typing import Any, Dict
import json
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torchmetrics
import yaml
import wandb

from models.pl_convlstm import ConvLSTMModule
from datamodules.TWSA_datamodule2D import TWSA_Datamodule2D
from utils.utils import read_config, save_config


def generate_trainer(config: Dict[str, Any]) -> pl.Trainer:
    """Generate a pytorch lightning trainer."""
    loggers = [pl.loggers.WandbLogger(
                    name=config["experiment"]["save_dir"].split("/")[-1],
                    save_dir=config["experiment"]["save_dir"],
                    project=config["wandb"]["project"],
                    resume="allow",
                    config=config,
                    mode=config["wandb"].get("mode", "online"))
                ]
    
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
    
    mask = datamodule.data_pred.land_ocean_mask[0] & datamodule.data_pred.mask_not_antarctica[0] # [SeqLen, H, W]
    f, (ax0, ax1, ax2, ax3) = plt.subplots(4,1,figsize=(5,10))
    vmax=40
    p0 = ax0.imshow(y_last, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    p1 = ax1.imshow(y_hat, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    p2 = ax2.imshow(y_true, cmap='coolwarm', vmin=-vmax, vmax=vmax )
    ape = (abs((y_hat-y_true)/(y_true+ 1e-10)))*mask *100
    p3 = ax3.imshow(ape, cmap='Reds', vmax=100)
    
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





def plot_explained_variance(y_hat, y_true, datamodule, trainer=None):
   
    # flip the cmap colors and add color for below 0 values
    cmap = plt.cm.Reds.reversed()
    cmap.set_under('blue')
    exp_var = torchmetrics.ExplainedVariance(multioutput='raw_values')(y_hat, y_true)
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
   
    
    pl.seed_everything(42, workers=True)
    config = read_config(config_path)
    config = create_experiment_dir(config)

    # generate trainer
    trainer = generate_trainer(config)

    twsa_sequence_data = TWSA_Datamodule2D(**config['datamodule'])
    
    
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
    with open(os.path.join(config["experiment"]["save_dir"], "eval_on_val.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)
    
    
 # --------------------------------------------------------------------------------------------------------------
 # Plotting predictions
 # --------------------------------------------------------------------------------------------------------------
    
    # generate prediction of last sequence in validation time series
    preds, true, input = trainer.predict(model, 
                                        dataloaders=twsa_sequence_data.predict_dataloader(time_step_idx=100, 
                                                                                        period=twsa_sequence_data.train_val_test_periods['val']))[0]
    title = 'TWSA Prediction on Val Set'   
    for i in range(config['datamodule']['output_seq_len']):
        
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
