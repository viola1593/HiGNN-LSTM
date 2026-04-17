
import argparse
import glob
import json
from logging import config
from xml.parsers.expat import model
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from typing import Any, Dict
import xarray as xr
import yaml

from datamodules.TWSA_heterograph_datamodule import TWSA_HeteroGraphDatamodule
from losses.kge import KGE
from losses.utils_for_nans import mean_with_nans, median_with_nans
from models.pl_hmmodel import HMModule



TEST_BASIN_IDS = { # Basin name : Basin ID
                    "Amazon" : 6030007000, # SA
                    "Parana": 6030016970, # SA
                    "Mississippi": 7030047060, # NA
                    "Mackenzie": 8030009560, # NA
                    "Danube": 2030008490, # EU
                    "Nile": 1030034260, # AF
                    "Congo": 1030020040, # AF
                    "Orange": 1030015850, # AF
                    "Zambezi": 1030011660, # AF
                    "Euphrates": 2030073570, # ME
                    "Volga": 2030068680, # EU
                    "Yangtze": 4030009880, # AS
                    "Amur": 4030000880, # AS
                    "Ganges": 4030025450, # AS
                    "Indus": 4030033640, # AS
                    "Murray": 5030073410 # AUS
                  } 


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

    # early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_rmse", mode="min")
    loggers = [pl.loggers.WandbLogger(
                    name=f"evaluate_{config['experiment']['save_dir'].split('/')[-1]}",
                    save_dir=config["experiment"]["save_dir"],
                    project=config["wandb"]["project"],
                    # entity=config["wandb"]["entity"],
                    resume="allow",
                    config=config,
                    mode=config["wandb"].get("mode", "online"))
                    ]

    return pl.Trainer(
        **config["pl"],
        default_root_dir=config["experiment"]["save_dir"],
        #logger=loggers,
        #callbacks=[early_stopping_callback]
        )


def create_experiment_dir(config: Dict[str, Any], dataset: str="test") -> str:
    """Create experiment directory.

    Args:
        config: config file
        dataset: dataset name (test or grace)

    Returns:
        config with updated save_dir
    """
    #os.makedirs(config["experiment"]["exp_dir"], exist_ok=True)
    exp_dir_path = os.path.join(f"eval_on_full{dataset}", config["experiment"]["save_dir"])
    os.makedirs(exp_dir_path, exist_ok=True)
    # os.makedirs(os.path.join(exp_dir_path, "SSIM"), exist_ok=True)
    # os.makedirs(os.path.join(exp_dir_path, "ExpVar"), exist_ok=True)
    # os.makedirs(os.path.join(exp_dir_path, "KGE"), exist_ok=True)
    # os.makedirs(os.path.join(exp_dir_path, "BasinPlots"), exist_ok=True)
    # os.makedirs(os.path.join(exp_dir_path, "Predictions"), exist_ok=True)
    # os.makedirs(os.path.join(exp_dir_path, "NSE"), exist_ok=True)
    # os.makedirs(os.path.join(exp_dir_path, "RMSE"), exist_ok=True)
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

def plot_prediction(y_hat, y_true, datamodule, antarctica=None, start_date=None,
                    save_dir=None, title='TWSA Prediction (Test Set)'):
    """y_hat, y_true: [N, T]"""
    lon, lat = datamodule.data_pred.land[0][:,0], datamodule.data_pred.land[0][:,1]

    if antarctica is not None:
        antarctica_mask =  (~datamodule.get_antarctica_landmask())
        lon = lon[antarctica_mask]
        lat = lat[antarctica_mask]

    vmax = 40
    f, axes = plt.subplots(3,6,figsize=(12,5))
    for i in range(y_hat.shape[1]):
    # p0 = ax0.scatter(lon, lat, c=y_last, cmap='coolwarm', s=1, vmin=-vmax, vmax=vmax)
        p1 = axes[0,i].scatter(lon, lat, c=y_hat[:,i], cmap='coolwarm', s=1,vmin=-vmax, vmax=vmax)
        p2 = axes[1,i].scatter(lon, lat, c=y_true[:,i], cmap='coolwarm', s=1,vmin=-vmax, vmax=vmax)
        ape = abs(y_hat[:,i]-y_true[:,i])/(abs(y_true[:,i])+ 1e-5) * 100
        p3 = axes[2,i].scatter(lon, lat, c=ape, cmap='Reds',s=1, vmin=0, vmax=100)
        # suptitle with mean APE
        axes[2,i].set_xlabel(f"Mean APE: {ape.mean():.2f}%")
        axes[0,i].set_title((start_date + pd.DateOffset(months=i)).strftime('%b %Y'))
        
        # remove axes tick labels
        format_global_plot(axes[0,i], noticks_x=True, noticks_y=True)
        format_global_plot(axes[1,i], noticks_x=True, noticks_y=True)
        format_global_plot(axes[2,i], noticks_x=True, noticks_y=True)


    axes[0,0].set_ylabel('Prediction', fontsize=12)
    axes[1,0].set_ylabel('Target', fontsize=12)
    axes[2,0].set_ylabel('Difference [%]', fontsize=12)
    # f.suptitle(title)
    plt.tight_layout()
    # set colorbar label
     
    # plt.colorbar(p0, ax=ax0)

    if save_dir is not None:
        plt.savefig(save_dir, dpi=300)
        plt.close()

def format_global_plot(ax, noticks_x=False, noticks_y=False):
    ax.tick_params(axis='both', direction='in', labelsize=12)
    
    
    if noticks_x:
        ax.set_xticks([])
    else:
        ax.set_xticks([-90, 0, 90])
        ax.set_xlabel('Longitude')
       
    if noticks_y:
        ax.set_yticks([])
    else:
        ax.set_yticks([-30,  0, 30, 60])
        ax.set_ylabel('Latitude')

def plot_NSE_or_KGE(nse, datamodule, title='NSE', save_dir=None, vmin=0):
    antarctica_mask =  (~datamodule.get_antarctica_landmask())
    x, y = datamodule.data_pred.land[0][:,0][antarctica_mask], datamodule.data_pred.land[0][:,1][antarctica_mask]
    f, ax0 = plt.subplots(1,1, figsize=(5,4))
    # -0.41 everything below is worse than a mean prediction for each time step
    cmap = plt.cm.Reds.reversed()
    if vmin is not None:
        cmap.set_under('blue')
    p0 = ax0.scatter(x, y, c=nse, s=1, vmin=vmin, vmax=1, marker='.', cmap=cmap)
    format_global_plot(ax0)
    plt.colorbar(p0, ax=ax0, extend='min')
    #add mean as text
    finmsk = np.isfinite(nse)
    plt.figtext(0.5, 0.00, f'Mean {title}: {nse[finmsk].mean().item():.2f}', ha='center', fontsize=10)

    ax0.set_title(title) 
    plt.tight_layout()

    plt.savefig(save_dir, dpi=300)
    plt.close()


def plot_RMSE(rmse, datamodule, title='RMSE', save_dir=None, vmin=0, vmax=20):
    antarctica_mask =  (~datamodule.get_antarctica_landmask())
    x, y = datamodule.data_pred.land[0][:,0][antarctica_mask], datamodule.data_pred.land[0][:,1][antarctica_mask]
    f, ax0 = plt.subplots(1,1, figsize=(5,4))
    # -0.41 everything below is worse than a mean prediction for each time step
    if title == 'RMSE':
        cmap = plt.cm.Reds
    elif title == 'Skill Score':
        cmap = plt.cm.coolwarm
    else: cmap = plt.cm.Reds
    if vmax > 1:
        cmap.set_over('blue')
    if vmin<0:
        cmap.set_under('blue')
    p0 = ax0.scatter(x, y, c=rmse, s=1, vmin=vmin, vmax=vmax, marker='.', cmap=cmap)
    format_global_plot(ax0)
    if vmin<0:
        plt.colorbar(p0, ax=ax0, extend='min')
    if vmax>1:
        plt.colorbar(p0, ax=ax0, extend='max')
    plt.figtext(0.5, 0.00, f'Mean {title}: {rmse.mean().item():.2f}', ha='center', fontsize=10)

    ax0.set_title(title) 
    plt.tight_layout()

    plt.savefig(save_dir, dpi=300)
    plt.close()

def plot_basintwsa_timesteps(name, id, pred, true, fupeng,  ax, fontsize=14):
    ''' pred shape [B, T], true shape [B, T], fupeng shape [B, T]
    Colors sequences with gradient from light to dark blue based on lead time (1-6)
    '''
    # Font size control - adjust these variables to change all text sizes
    fontsize_title = fontsize+1
    fontsize_text = fontsize-2
    fontsize_ticks = fontsize
    
    x_base = np.arange(6)
    linewidth = 1.0
    # Colormap for lead time gradient
    lead_time_cmap = plt.cm.Blues.reversed()
    fup_lead_time_cmap = plt.cm.Reds.reversed()


    
    for seq_idx, (p, t, f) in enumerate(zip(pred, true, fupeng)):
        x = x_base + seq_idx  # Offset for each sequence
        
        if seq_idx == 0:
            ax.plot(x, t, label='GRACE-FO CSR RL06.3', color='black', marker='.', linewidth=1.5)
            # ax.plot(x, f, label='Climatology', linestyle='dotted', color='gray', linewidth=linewidth*2)
            # Add dummy line for legend
            # ax.plot([], [], color='royalblue', linewidth=linewidth, label='HiGNN-LSTM')
            # ax.plot([], [], color='darkred', linewidth=linewidth, label='Li et al.')
        else: 
            ax.plot(x, t, color='black', marker='.', linewidth=1.5)
            # ax.plot(x, c, linestyle='dotted', color='gray', linewidth=linewidth)
        
        # Plot prediction with gradient color based on lead time
        for j in range(len(p) - 1):
            # Color based on lead time position (1-6 normalized to 0-1)
            color = lead_time_cmap((j + 1) / 6.0)
            # Plot segment from lead time j to j+1
            ax.plot(x[j:j+2], p[j:j+2], color=color, linewidth=linewidth)
        for j in range(len(f) - 1):
            # Color based on lead time position (1-6 normalized to 0-1)
            color = fup_lead_time_cmap((j + 1) / 6.0)
            # Plot segment from lead time j to j+1
            ax.plot(x[j:j+2], f[j:j+2], color=color, linewidth=linewidth)
        
        # Plot last point
        ax.plot(x[-1], p[-1], color=lead_time_cmap(1.0), linewidth=linewidth)
        ax.plot(x[-1], f[-1], color=fup_lead_time_cmap(1.0), linewidth=linewidth)
    
    # Calculate metrics
    
    pred_tensor = torch.tensor(pred, dtype=torch.float32).squeeze().permute(1,0) # [B, T]
    true_tensor = torch.tensor(true, dtype=torch.float32).squeeze().permute(1,0)
    fupeng_tensor = torch.tensor(fupeng, dtype=torch.float32).squeeze().permute(1,0)
    _corr = torchmetrics.PearsonCorrCoef(num_outputs=pred.shape[0])
    rmse_pred = _rmse(pred_tensor, true_tensor).mean()
    corr_pred = _corr(pred_tensor, true_tensor).mean()
    rmse_fup = _rmse(fupeng_tensor, true_tensor).mean()
    corr_fup = _corr(fupeng_tensor, true_tensor).mean()
    
    # Add metrics text on the plot
    metrics_text = (f'r={corr_pred:.2f}/{corr_fup:.2f}\n'
                    f'RMSE={rmse_pred:.2f}/{rmse_fup:.2f} cm')
    ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes, fontsize=fontsize_text,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(f'{name}', fontsize=fontsize_title)
    # ax.set_title(f'{name} ({id})', fontsize=fontsize_title)

    # ax.set_xlabel('Forecast Step', fontsize=11)
    # ax.set_ylabel('TWSA (cm)', fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    ax.grid(True, which='major', axis='both', linestyle='--', color='gray', alpha=0.7)
   
    return ax

def plot_basintwsa(name, id, fupeng, pred, true, seasonal, ax=None):
    ''' pred shape [B, T], true shape [B, T], seasonal shape [B, T] '''
    x = np.arange(pred.shape[0])
    pred = np.transpose(pred[:,:,0])
    fupeng = np.transpose(fupeng[:,:,0])
    true = np.transpose(true[:,:,0])
    seasonal = np.transpose(seasonal[:,:,0])
    linewidth = 1
    cmap = 'Reds'
    cmapgnn = plt.cm.Reds
    cmapfup = plt.cm.Blues
    colors_gnn = [cmapgnn(i / 10) for i in range(10)]
    colors_fup = [cmapfup(i /10) for i in range(10)]
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,8))
    
    ax.plot(x, seasonal[0], label='Seasonal Trend', linestyle='dotted', color='gray', linewidth=linewidth*2)

    for i, (f, p, t) in enumerate(zip(fupeng, pred, true)):
        if i ==0:
            ax.plot(x, t, label='GRACE-FO CSR RL06.3 ', color='black', marker='o')
            ax.plot(x, p, label='HiGNN', color = colors_gnn[i+4],  linewidth=linewidth)
            ax.plot(x, f, label='Li et al.', color = colors_fup[i+4],  linewidth=linewidth)
        else:
            i = i % 6
            ax.plot(x, t,  color='black', marker='o')
            ax.plot(x, p, color = colors_gnn[i+4], linewidth=linewidth, )
            #ax.plot(x, c, linestyle='dotted', color='gray', linewidth=linewidth*2)
            ax.plot(x, f, color = colors_fup[i+4],  linewidth=linewidth)
        x+=1

    # Calculate metrics
    pred_flat = np.concatenate(pred)
    fupeng_flat = np.concatenate(fupeng)
    true_flat = np.concatenate(true)
    
    pred_tensor = torch.tensor(pred_flat, dtype=torch.float32)
    true_tensor = torch.tensor(true_flat, dtype=torch.float32)
    fupeng_tensor = torch.tensor(fupeng_flat, dtype=torch.float32)
    
    rmse_pred = _rmse(pred_tensor.flatten(), true_tensor.flatten()).item()
    corr_pred = _corr(pred_tensor.flatten(), true_tensor.flatten()).item()
    rmse_fup = _rmse(fupeng_tensor.flatten(), true_tensor.flatten()).item()
    corr_fup = _corr(fupeng_tensor.flatten(), true_tensor.flatten()).item()
    
    # Add metrics text on the plot
    metrics_text = (f'HiGNN: RMSE={rmse_pred:.2f} cm, r={corr_pred:.3f}\n'
                    f'Li et al.: RMSE={rmse_fup:.2f} cm, r={corr_fup:.3f}')
    ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(f'{name} ({id})', fontsize=13)
    ax.set_ylabel('TWSA (cm)', fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, which='both', axis='both', linestyle='--', color='gray', alpha=0.7)

    return ax


def plot_leadtimes_meantwsa(fupeng, pred, true, save_dir=None):
    ''' pred shape [B, T], true shape [B, T], clim shape [B, T] '''
    x = np.arange(pred.shape[0])
    pred = np.transpose(pred)
    fupeng = np.transpose(fupeng)
    true = np.transpose(true)
    linewidth = 1


    fig, ax = plt.subplots(2,3,figsize=(10,8))
    ax = ax.flatten()
    for i, (f, p, t) in enumerate(zip(fupeng, pred, true)):
        if i ==0:
            ax[i].plot(x, t, label='True', color='black', marker='o')
            ax[i].plot(x, p, label='HiGNN-LSTM', color = 'b',  linewidth=linewidth)
            #ax.plot(x, c, label='Climatology', linestyle='dotted', color='gray', linewidth=linewidth*2)
            ax[i].plot(x, f, label='Fupeng et al.', color = 'r',  linewidth=linewidth)
        else:
            i = i % 6
            ax[i].plot(x, t,  color='black', marker='o')
            ax[i].plot(x, p, color = 'b', linewidth=linewidth, )
            #ax.plot(x, c, linestyle='dotted', color='gray', linewidth=linewidth*2)
            ax[i].plot(x, f, color = 'r',  linewidth=linewidth)
        ax[i].grid(True, which='both', axis='both', linestyle='--', color='gray', alpha=0.7)
        ax[i].set_title(f'{i+1} months lead')
        x+=1
    ax[0].legend(loc='upper right')
    plt.tight_layout()

    plt.savefig(save_dir, dpi=300)
    plt.close()



def build_model_from_checkpoint(experiment_path, config, gpu, twsa_sequence_data):
    # Recursively search for .ckpt files in all subfolders
    ckpt_files = glob.glob(os.path.join(experiment_path, "**", "*.ckpt"), recursive=True)
    
    if ckpt_files:
        # Use the latest checkpoint (by modification time)
        latest_ckpt = min(ckpt_files, key=os.path.getmtime)
        print(f"Found checkpoint: {latest_ckpt}")
    else:
        print("No checkpoint found.")
        return None
    #load model from checkpoint
    model = HMModule.load_from_checkpoint(latest_ckpt, **config["model"], 
                                          num_input_features=twsa_sequence_data.num_features,
                                          num_ocean_features=twsa_sequence_data.num_ocean_features,
                                            sequence_length=twsa_sequence_data.input_seq_len,
                                            output_seq_len=twsa_sequence_data.target_seq_len, 
                                            land_sea_mask=None,
                                            map_location=f'cuda:{gpu}')
   

    model.model_summary()
    return model

def get_predictions(twsa_sequence_data, trainer, config, dataset='test', gpu=0):
    if dataset == "test":
        dataloader_pred = twsa_sequence_data.predict_dataloader(period=twsa_sequence_data.train_val_test_periods['test'])
        # dataloader_pred = twsa_sequence_data.predict_dataloader(period=twsa_sequence_data.train_val_test_periods['grace'])
    elif dataset == "val":
        dataloader_pred = twsa_sequence_data.predict_dataloader(period=twsa_sequence_data.train_val_test_periods['val'])
        # dataloader_pred = twsa_sequence_data.predict_dataloader(period=twsa_sequence_data.train_val_test_periods['grace'])
    elif dataset == "train":
        dataloader_pred = twsa_sequence_data.predict_dataloader(period=twsa_sequence_data.train_val_test_periods['train'])

    elif dataset == "grace":
        dataloader_pred = twsa_sequence_data.GRACE_dataloader()
    
    try:
        predictions = torch.load(os.path.join(config["experiment"]["save_dir"],"predictions.pt"))
    except FileNotFoundError:
        model = build_model_from_checkpoint(config["experiment"]["save_dir"], config, gpu, twsa_sequence_data)

        trainer.test(model, twsa_sequence_data)
        # test before prediction important so trainer has access to datamodules unnormalization during prediction

        predictions = trainer.predict(model, dataloader_pred) # predict usually selects val time period
        torch.save(predictions, os.path.join(config["experiment"]["save_dir"],"predictions.pt"))
    return predictions


def recover_month_index(month_sin, month_cos):
    # shape: [B, T]
    angle = torch.atan2(month_sin, month_cos)  # range: [-π, π]
    month_float = (angle % (2 * np.pi)) * 12 / (2 * np.pi)  # range: [0, 12)
    month_index = month_float.long()  # round down to nearest integer (0–11)
    return month_index


def compute_climatology(data: xr.DataArray, lsm):
    # y_true: [B, N, T], months: [B, T]
    """Compute climatology from the dataset."""
    climatology = data.groupby('time.month').mean(dim='time').values  # [12, N]
    land_clim = climatology[lsm[:12]].reshape(12, -1)

    return land_clim  # [12, Num land nodes]

def get_climatology(twsa_sequence_data, config):
    # compute climatology from training data
    try: 
        climatology = torch.load(os.path.join(config["experiment"]["save_dir"],"climatology.pt"))
    except FileNotFoundError:
        twsa_train = twsa_sequence_data.unnormalize_twsa(twsa_sequence_data.data_train.ds_timeslice['twsa'])
        climatology = compute_climatology(twsa_train, 
                                            twsa_sequence_data.data_train.land_ocean_mask)
        torch.save(climatology, os.path.join(config["experiment"]["save_dir"],"climatology.pt"))
    return climatology


def basin_wide_mean_twsa(y_hat, y_true, climatology, datamodule):
    """Calculate the basin wide mean TWSA predictions/ground truth per each time step.
    Should be 238 (includes 1 dummy basin for hawaii) basins in total.
    Args:
        y_hat: predicted values, shape [T, N]
        y_true: true values, shape [T, N]
        datamodule: datamodule containing the land_ocean_mask
    returns:
        pred_basin_means: predicted basin means, shape [T, N_basins]
        true_basin_means: true basin means, shape [T, N_basins]
    """
    
    antarctica_mask =  (~datamodule.get_antarctica_landmask())
    lon, lat = datamodule.data_pred.land[0][:,0], datamodule.data_pred.land[0][:,1]
    land2basin = datamodule.data_pred.land_to_basin_ids
    pred_basin_means = []
    true_basin_means = []
    climatology_mean = []
    for pred, true, clim in zip(y_hat, y_true, climatology):
        df = pd.DataFrame({
            'lon': lon[antarctica_mask],
            'lat': lat[antarctica_mask],
            'twsa_pred': pred,
            'twsa_true': true,
            'climatology': clim,
            'basin': land2basin[antarctica_mask]
            })
        groups = df.groupby('basin').mean()
        
        pred_basin_means.append(groups['twsa_pred'].values)
        true_basin_means.append(groups['twsa_true'].values)
        climatology_mean.append(groups['climatology'].values)
    pred_basin_means = np.stack(pred_basin_means)  # [T, N]
    true_basin_means = np.stack(true_basin_means)  # [T, N]
    climatology_mean = np.stack(climatology_mean)  # [T, N]
    return pred_basin_means, true_basin_means, climatology_mean, groups.index


def _mape(y_hat, y_true):
    """Calculate Mean Absolute Percentage Error (MAPE) between two tensors."""
    # Avoid division by zero
    epsilon = 1e-8
    return torch.nanmean(torch.abs((y_true - y_hat) / (y_true + epsilon)), dim=1) * 100  # in percentage

def _rmse(y_hat, y_true):
    """Calculate Root Mean Squared Error (RMSE) between two tensors."""
    return torch.sqrt(torch.nanmean((y_true - y_hat) ** 2, dim=1))  # RMSE

def _lat_weighted_rmse(y_hat, y_true, lat_weigths):
    """Calculate Root Mean Squared Error (RMSE) between two tensors."""
    return torch.sqrt(torch.nanmean((y_true - y_hat) ** 2 * lat_weigths[None, :,None], dim=1))  # RMSE

def _smape(y_hat, y_true):
    """Calculate Symmetric Mean Absolute Percentage Error (SMAPE) between two tensors."""
    # Avoid division by zero
    epsilon = 1e-8
    return torch.nanmean(torch.abs(y_true - y_hat) / (torch.abs(y_true) + torch.abs(y_hat) + epsilon), dim=1) * 100  # in percentage

def _corr(y_hat, y_true):
    pearson = torchmetrics.PearsonCorrCoef()
    return pearson(y_hat, y_true)

def nse(y_hat, y_true):
    """Calculate Nash-Sutcliffe Efficiency (NSE) with long term mean (climatology) or ground truth.
    y_hat: shape [T (time steps), N (num nodes)]
    y_true: shape [T, N]"""
    numerator = torch.sum((y_true - y_hat) ** 2, dim=0)
    denominator = torch.sum((y_true - torch.mean(y_true, dim=0, keepdim=True)) ** 2, dim=0)
    return 1 - numerator / (denominator + 1e-8)

def calculate_per_timestep_scores(y_hat, y_true, climatology, scores_dict, metrics_dict):
    """Calculate per timestep scores for the given predictions and ground truth.
    Args: 
        y_hat: shape [time steps, num nodes]
        y_true: shape [time steps, num nodes]
        climatology: shape [time steps, num nodes]"""
    
    y_hat_deseas = y_hat - climatology
    y_true_deseas = y_true - climatology
    
    for metric_name, metric_func in metrics_dict.items():
        if metric_name == 'exp_var':
            # time step as last dim
            y_hat_new = y_hat.transpose(1,0)
            y_true_new  = y_true.transpose(1,0)
            y_hat_deseas_new  = y_hat_deseas.transpose(1,0)
            y_true_deseas_new  = y_true_deseas.transpose(1,0)
            climatology_new  = climatology.transpose(1,0)
            
            # Calculate scores for each time step
            scores = metric_func(y_hat_new, y_true_new)
            # Calculate deseasonalized scores
            deseas_scores = metric_func(y_hat_deseas_new, y_true_deseas_new)
            # Calculate climatology scores
            clim_scores = metric_func(torch.Tensor(climatology_new), y_true_new)
        
        else:
            # Calculate scores for each time step
            scores = metric_func(y_hat, y_true)
            # Calculate deseasonalized scores
            deseas_scores = metric_func(y_hat_deseas, y_true_deseas)
            # Calculate climatology scores
            clim_scores = metric_func(torch.Tensor(climatology), y_true)
        
        scores_dict[f'{metric_name}'].append(scores.cpu().numpy())
        scores_dict[f'{metric_name}_deseas'].append(deseas_scores.cpu().numpy())
        scores_dict[f'{metric_name}_clim'].append(clim_scores.cpu().numpy())
    return scores_dict

def get_months_output_sequences(months_input_sequences, output_seq_len):
    months_output_sequences = [(months_input_sequences[:,:, -1] +i +1)%12 for i in range(output_seq_len)]
    months_output_sequences = torch.stack([(months_input_sequences[:,:, -1] +i +1)%12 for i in range(output_seq_len)])  # [horizon, Total_B, N]
    months_output_sequences = months_output_sequences.permute(1,2,0)  # [Total_B, N, horizon]
    return months_output_sequences  

def get_basin_wide_mean_twsa(twsa_pred, twsa_true, climatology, months_output_sequences, datamodule):
    # calculate basin-wide mean TWSA for each prediction sequence
    pred_basin_mean = []
    true_basin_mean = []
    basin_mean_climatology = []
    pred_month = []
    
    for (pred, true, months) in zip(twsa_pred, twsa_true, months_output_sequences):
        pred, true, basin_clim, basin_ids = basin_wide_mean_twsa(pred.permute(1,0),
                                                                  true.permute(1,0), 
                                                                  climatology[months[0]], 
                                                                   datamodule)
        pred_basin_mean.append(pred)
        true_basin_mean.append(true)
        basin_mean_climatology.append(basin_clim)
        pred_month.append(months[0])
    pred_basin_mean = np.stack(pred_basin_mean)  # [B, T (horizon), N_basins]
    true_basin_mean = np.stack(true_basin_mean)  # [B, T(horizon), N_basins]
    basin_mean_climatology = np.stack(basin_mean_climatology)  # [B, T(horizon), N_basins]
    pred_months = np.stack(pred_month)  # [B, T(horizon)]
    
    return pred_basin_mean, true_basin_mean, basin_mean_climatology, pred_months, basin_ids


def evaluate(experiment_path, gpu, dataset="test"):
    """Evaluation Script.
    Args:
        experiment_path: path to the experiment directory
        gpu: gpu id to use
        dataset: dataset name (test or grace)
    """
    # torch.set_float32_matmul_precision("medium")
    
    pl.seed_everything(42, workers=True)
    config = read_config(os.path.join(experiment_path,  'config.yaml'))
    # exchange devices to given gpu in config
    config['pl']['devices'] = [gpu]

    config = create_experiment_dir(config, dataset=dataset)

    # generate trainer
    trainer = generate_trainer(config)

    twsa_sequence_data = TWSA_HeteroGraphDatamodule(**config['datamodule'])
    twsa_sequence_data.setup(stage='fit')
    



    # --------------------------------------------------------------------------------------------------------------
    # Preparing predictions, ground truth, and climatology for evaluations and plotting
    # --------------------------------------------------------------------------------------------------------------

    climatology = get_climatology(twsa_sequence_data, config)
     # get list of length num_batches, each element is a tuple of (y_hat, y_true, input_features)
    predictions = get_predictions(twsa_sequence_data, trainer, config, dataset=dataset, gpu=gpu)
    time_period = twsa_sequence_data.data_pred.time_period

    sample_indices = twsa_sequence_data.data_pred.samples
    timesteps = twsa_sequence_data.data_pred.ds_timeslice.time.values[sample_indices]

    num_nodes = twsa_sequence_data.data_pred.num_land_nodes() 
    # get mask to mask antarctica out in the evaluation
    antarctica_mask =  (~twsa_sequence_data.get_antarctica_landmask())

    
    # iterate over the batches in the predictions for the evaluations
    output_seq_len = config['datamodule']['output_seq_len']
    twsa_pred = torch.cat([batch[0].reshape(-1, num_nodes, output_seq_len) for batch in predictions], dim=0)[:, antarctica_mask] # [Total_B, N, T]
    twsa_true = torch.cat([batch[1].reshape(-1, num_nodes, output_seq_len) for batch in predictions], dim=0)[:, antarctica_mask] # [Total_B, N, T]
    months_input_sequences = torch.cat([recover_month_index(batch[2]['grid'].x[:, :,-1], batch[2]['grid'].x[:, :,-2]).reshape(-1, num_nodes, 12) for batch in predictions], dim=0)[:,antarctica_mask]  # [Total_B, N, T]
    months_output_sequences = get_months_output_sequences(months_input_sequences, output_seq_len)  # [Total_B, N, T]
    # mask antarctica out of climatology as well
    climatology = climatology[:,antarctica_mask]  # [12, N]
    
    
   


    
    
# --------------------------------------------------------------------------------------------------------------------------------------   
# Prepare data from Fupeng et al. for comparison with our model predictions
# -------------------------------------------------------------

    file_path = "data/fupeng_csr_fcast/combined.nc" # period between nov 2019 - nov 2023
    dataset = xr.open_dataset(file_path)
    land_sea_mask = twsa_sequence_data.land_ocean_mask
    land_sea_mask = land_sea_mask[:6]
    
   
    fupeng_ds = [dataset[ts].values[land_sea_mask].reshape(6,-1)[:, antarctica_mask] for ts in dataset][:-6] #last five timesteps are too much (accident data downloading)
    fupeng_ds = torch.Tensor(np.stack(fupeng_ds)).permute(0,2,1) # shape num_pred_seqs, nodes, seq_len


    # crop our prediction to fit the time steps of the Li et al. forecast, which starts later than ours (Nov 2019 )
    starting_idx = 17 # Nov 2019 in our dataset
    # li_timesteps = fupeng_ds.shape[0]
    # starting_timestep = twsa_pred.shape[0]-li_timesteps
    twsa_pred = twsa_pred[starting_idx:]
    twsa_true = twsa_true[starting_idx:]
    timesteps = timesteps[starting_idx:]

    pred_basin_mean, true_basin_mean, basin_mean_climatology, pred_months, basin_ids = get_basin_wide_mean_twsa(twsa_pred, twsa_true, climatology, months_output_sequences, twsa_sequence_data)





    fupeng_pred_basin_mean, _, _, _, _ = get_basin_wide_mean_twsa(torch.nan_to_num(fupeng_ds), twsa_true, climatology, months_output_sequences, twsa_sequence_data)

    # do the same for the seasonal trends from Li et al. 
    file_path = "data/fupeng_csr_fcast/LiSeasonal.nc" # period between nov 2019 - nov 2023
    seas_dataset = xr.open_dataset(file_path)
 
   
    seasonal_ds = [seas_dataset[ts].values[land_sea_mask].reshape(6,-1)[:, antarctica_mask] for ts in seas_dataset][:-6] #last six timesteps are too much 
    seasonal_ds = torch.Tensor(np.stack(seasonal_ds)).permute(0,2,1) # shape num_pred_seqs, nodes, seq_len

    seasonal_basin_mean, _, _, _, _ = get_basin_wide_mean_twsa(torch.nan_to_num(seasonal_ds), twsa_true, climatology, months_output_sequences, twsa_sequence_data)

# --------------------------------------------------------------------------------------------------------------------------------------   
# calculate metrics - comparison with GRACE from  CSR RL06.03
# --------------------------------------------------------------------------------------------------------------------------------------   
    
    metrics_dict = {'rmse': _rmse, 'mape': _mape, 'smape': _smape}
    # get climatology for the predicted months
    clim_pred = torch.Tensor(climatology[pred_months])
                    
    # per_timestep_scores = [{f'{name}': metric(twsa_pred, twsa_true).mean(dim=0) for name, metric in metrics_dict.items()}][0]
    clim_per_timestep_scores = [{f'clim_{name}': metric(clim_pred.permute(0, 2, 1), twsa_true).mean(dim=0) for name, metric in metrics_dict.items()}][0]
    # basin_scores = [{f'{name}': metric(torch.Tensor(pred_basin_mean), torch.Tensor(true_basin_mean)).mean(dim=0) for name, metric in metrics_dict.items()}][0]
    # basin_per_timestep_scores = [{f'{name}': metric(torch.Tensor(pred_basin_mean).permute(2, 0, 1), torch.Tensor(true_basin_mean).permute(2, 0, 1)).mean(dim=0) for name, metric in metrics_dict.items()}][0]

    fupeng_per_timestep_scores = [{f'{name}': metric(fupeng_ds, twsa_true).mean(dim=0) for name, metric in metrics_dict.items()}][0]
    fupeng_basin_scores = [{f'{name}': metric(torch.Tensor(fupeng_pred_basin_mean), torch.Tensor(true_basin_mean)).mean(dim=0) for name, metric in metrics_dict.items()}][0]
    fupeng_basin_per_timestep_scores = [{f'{name}': metric(torch.Tensor(fupeng_pred_basin_mean).permute(2, 0, 1), torch.Tensor(true_basin_mean).permute(2, 0, 1)).mean(dim=0) for name, metric in metrics_dict.items()}][0]

    gnn_per_timestep_scores = [{f'gnn_{name}': metric(twsa_pred, twsa_true).mean(dim=0) for name, metric in metrics_dict.items()}][0]


    # save scores to csv
    per_timestep_df = pd.DataFrame(fupeng_per_timestep_scores, index=np.arange(6))
    per_timestep_df = per_timestep_df.assign(**clim_per_timestep_scores)
    per_timestep_df = per_timestep_df.assign(**gnn_per_timestep_scores)
    per_timestep_df.index.name = 'Timestep'
    per_timestep_df.to_csv(os.path.join(config["experiment"]["save_dir"],"fupengper_timestep_scores.csv"))
    basin_df = pd.DataFrame(fupeng_basin_scores, index=basin_ids)
    basin_df.index.name = 'Basin ID'
    basin_df.to_csv(os.path.join(config["experiment"]["save_dir"],"fupeng_basin_scores.csv"))
    basin_per_timestep_df = pd.DataFrame(fupeng_basin_per_timestep_scores, index=np.arange(6))
    basin_per_timestep_df.index.name = 'Timestep'
    basin_per_timestep_df.to_csv(os.path.join(config["experiment"]["save_dir"],"fupeng_basin_per_timestep_scores.csv"))
# --------------------------------------------------------------------------------------------------------------------------------------   
# calculate latitude weighted metrics - comparison with GRACE from  CSR RL06.03
# --------------------------------------------------------------------------------------------------------------------------------------   

    lat_weights = torch.cos(torch.deg2rad(torch.Tensor(twsa_sequence_data.data_train.land[0][:,1][antarctica_mask])))  # [N]
    # normalize so the weights sum to 1 (avoid changing overall loss scale)
    lat_weights = lat_weights / (lat_weights.sum() + 1e-12)
    lat_weighted_rmse = _lat_weighted_rmse(twsa_pred, twsa_true, lat_weights)
    lat_weighted_fupeng_rmse = _lat_weighted_rmse(fupeng_ds, twsa_true, lat_weights)
    print(f'Latitude weighted RMSE for HiGNN: {lat_weighted_rmse.mean().item():.2f} cm')
    print(f'Latitude weighted RMSE for Fupeng et al.: {lat_weighted_fupeng_rmse.mean().item():.2f} cm')

# --------------------------------------------------------------------------------------------------------------------------------------   
# plot global mean 
# ------------------------------------------------------------- 
    
    plot_leadtimes_meantwsa(torch.nanmean(fupeng_ds, dim=1), 
                                 torch.mean(twsa_pred, dim=1), 
                                 torch.mean(twsa_true, dim=1),
                                 save_dir=os.path.join(config["experiment"]["save_dir"], f'GlobalMean_FupvsGNN.png'))



# --------------------------------------------------------------------------------------------------------------------------------------   
# Plotting basin means - comparison
# ------------------------------------------------------------- 
    
    
    # Plotting basin means 
    months_to_plot = 48
    # end_year = twsa_sequence_data.data_pred.ds_timeslice.time.values[-1]
    # start_year = end_year - pd.DateOffset(months=months_to_plot-1)
    start_year = timesteps[0]

    # end_year = pd.to_datetime(time_period[1])
    fig, axes = plt.subplots(4, 4, figsize=(10,10), sharex=True)
    fontsize = 12
    axes_flat = axes.flatten()
    GRACE_BASIN_IDS = { # Basin name : Basin ID
                    "Amazon" : 6030007000, # SA
                    "Parana": 6030016970, # SA
                    "Mississippi": 7030047060, # NA
                    "Mackenzie": 8030009560, # NA
                    "Danube": 2030008490, # EU
                    "Nile": 1030034260, # AF
                    "Congo": 1030020040, # AF
                    "Orange": 1030015850, # AF
                    "Zambezi": 1030011660, # AF
                    "Euphrates": 2030073570, # ME
                    "Volga": 2030068680, # EU
                    "Yangtze": 4030009880, # AS
                    "Amur": 4030000880, # AS
                    "Ganges": 4030025450, # AS
                    "Indus": 4030033640, # AS
                    "Murray": 5030073410 # AUS
                  } # only take basin with strong variability 


    
    for i, (name, id) in enumerate(GRACE_BASIN_IDS.items()):
        ax = axes_flat[i]
        pred = pred_basin_mean[:,:,basin_ids == id] # shape (B, T)
        true = true_basin_mean[:,:,basin_ids == id]
        fupeng = fupeng_pred_basin_mean[:,:,basin_ids == id]
        seas = seasonal_basin_mean[:,:,basin_ids == id]
        plot_basintwsa_timesteps(name, id, pred, true, fupeng, 
                                 ax=ax, fontsize=fontsize)

                                 
        # start x labels from +1 instead of 0 because of 12 months input 
        
        if i > 1:
            x_labels = [(start_year + pd.DateOffset(years=i)).strftime('%b %Y') for i in range(((len(pred)+6)//12+1))]
            ax.set_xticks(list(np.arange(0,len(pred)+6+1, 12)))
            ax.set_xticklabels(x_labels,  rotation=90)
        else: 
            ax.set_xticklabels([])
        if i in [0,4,8,12]:
            ax.set_ylabel('TWSA [cm]', fontsize=fontsize)
    # ax.legend(loc='upper left', fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    legend_ax = fig.add_axes([0.55, 0.05, 0.2, 0.02]) 
    legend_ax.axis("off")
    legend_ax.legend(handles, labels,
            loc="right",
            fontsize=fontsize-2)
    text_ax = fig.add_axes([0.76, 0.05, 0.2, 0.02]) 
    text_ax.axis("off")
    text_ax.text( 0.0, 0.5,                          # x=left-aligned, y=vertically centered in axes
            'r / RMSE = HiGNN-LSTM / Li et al.',
            fontsize=fontsize-2,
            horizontalalignment='left',
            verticalalignment='center',        # center within the 0.02-height axes
            transform=text_ax.transAxes,       # coordinates relative to this axes
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
  

    # fig.subplots_adjust(bottom=0.08) 
    lead_time_cmap = plt.cm.Blues.reversed() 
    fup_lead_time_cmap = plt.cm.Reds.reversed() 
    sm = plt.cm.ScalarMappable(cmap=lead_time_cmap, norm=plt.Normalize(vmin=1, vmax=6))
    sm.set_array([])
    sm2 = plt.cm.ScalarMappable(cmap=fup_lead_time_cmap, norm=plt.Normalize(vmin=1, vmax=6))
    sm2.set_array([])
    cbar_ax = fig.add_axes([0.05, 0.05, 0.2, 0.02])
    # cbar_ax = fig.add_axes([0.15, 0.05, 0.3, 0.02])   # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('HiGNN-LSTM lead time (months)', fontsize=fontsize-2)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_ticks([1, 2, 3, 4, 5, 6])
    cbar.ax.tick_params(which='minor', length=0)
    cbar_ax2 = fig.add_axes([0.3, 0.05, 0.2, 0.02]) 
    # bar_ax2 = fig.add_axes([0.55, 0.05, 0.3, 0.02]) 
    cbar2 = fig.colorbar(sm2, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label('Li et al. lead time (months)', fontsize=fontsize-2)
    cbar2.ax.tick_params(labelsize=fontsize)
    cbar2.set_ticks([1, 2, 3, 4, 5, 6])
    cbar2.ax.tick_params(which='minor', length=0)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    
    plt.savefig(os.path.join(config["experiment"]["save_dir"], 
                                                      f'BasinPlots/Fupengcomp_lessBasins_leadwise.png'), dpi=300)
        # test_basin_per_timestep_scores = [{f'{name}': metric(torch.Tensor(pred), torch.Tensor(true)).mean(dim=0) for name, metric in metrics_dict.items()}][0]

         
   
    # save config
    save_config(config, os.path.join(config["experiment"]["save_dir"], "config.yaml"))





def start():
    """Start training."""
    # Command line arguments
    parser = argparse.ArgumentParser(
        prog="baseline_evaluation.py", description="Evaluates a trained model given the path to its config and chkpts."
    )

    parser.add_argument("--exp_path", help="Path to the experiment directory", required=True)
    parser.add_argument(
        "--gpu", help="GPU device ID", default=0, type=int,
    )
    parser.add_argument(
        "--dataset", help="Dataset to evaluate on (test, val, or grace)", default="test", type=str)

    args = parser.parse_args()

    evaluate(args.exp_path, gpu=args.gpu, dataset=args.dataset)


if __name__ == "__main__":
    start()
