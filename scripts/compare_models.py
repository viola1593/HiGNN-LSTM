
import argparse
import glob
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
from models.pl_hmmodel import HMModule
from utils.utils import read_config, save_config



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





def generate_trainer(config: Dict[str, Any]) -> pl.Trainer:
    """Generate a pytorch lightning trainer."""

    return pl.Trainer(
        **config["pl"],
        default_root_dir=config["experiment"]["save_dir"],
        )


def create_experiment_dir(config: Dict[str, Any], dataset: str="test") -> str:
    """Create experiment directory.

    Args:
        config: config file
        dataset: dataset name (test or grace)

    Returns:
        config with updated save_dir
    """
    exp_dir_path = os.path.join(f"eval_on_full{dataset}", config["experiment"]["save_dir"])
    os.makedirs(exp_dir_path, exist_ok=True)
    config["experiment"]["save_dir"] = exp_dir_path
    
    return config










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

        else: 
            ax.plot(x, t, color='black', marker='.', linewidth=1.5)

        
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
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    ax.grid(True, which='major', axis='both', linestyle='--', color='gray', alpha=0.7)
   
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
    elif dataset == "val":
        dataloader_pred = twsa_sequence_data.predict_dataloader(period=twsa_sequence_data.train_val_test_periods['val'])
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


def evaluate(experiment_path, gpu, dataset="test", 
             FCast_file_path="data/fupeng_csr_fcast/combined.nc"):
    """Evaluation Script.
    Args:
        experiment_path: path to the experiment directory
        gpu: gpu id to use
        dataset: dataset name (test or grace)
    """

    
    pl.seed_everything(42, workers=True)
    config = read_config(os.path.join(experiment_path,  'config.yaml'))
    # exchange devices to given gpu in config
    config['pl']['devices'] = [gpu]

    config = create_experiment_dir(config, dataset=dataset)

    # generate trainer
    trainer = generate_trainer(config)

    twsa_sequence_data = TWSA_HeteroGraphDatamodule(**config['datamodule'])
    twsa_sequence_data.setup(stage='fit')
    trainer.datamodule = twsa_sequence_data # assign datamodule to trainer for prediction
    



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
# Prepare data from Li et al. for comparison with our model predictions
# -------------------------------------------------------------

    dataset = xr.open_dataset(FCast_file_path) # period between nov 2019 - nov 2023
    land_sea_mask = twsa_sequence_data.land_ocean_mask
    land_sea_mask = land_sea_mask[:6]
    
   
    fcast = [dataset[ts].values[land_sea_mask].reshape(6,-1)[:, antarctica_mask] for ts in dataset][:-6] #last five timesteps are too much (accident data downloading)
    fcast = torch.Tensor(np.stack(fcast)).permute(0,2,1) # shape num_pred_seqs, nodes, seq_len


    # crop our prediction to fit the time steps of the Li et al. forecast, which starts later than ours (Nov 2019 )
    starting_idx = 17 # Nov 2019 in our dataset

    twsa_pred = twsa_pred[starting_idx:]
    twsa_true = twsa_true[starting_idx:]
    timesteps = timesteps[starting_idx:]

    pred_basin_mean, true_basin_mean, basin_mean_climatology, pred_months, basin_ids = get_basin_wide_mean_twsa(twsa_pred, twsa_true, climatology, months_output_sequences, twsa_sequence_data)



    fcast_pred_basin_mean, _, _, _, _ = get_basin_wide_mean_twsa(torch.nan_to_num(fcast), twsa_true, climatology, months_output_sequences, twsa_sequence_data)


# --------------------------------------------------------------------------------------------------------------------------------------   
# calculate metrics - comparison with GRACE from  CSR RL06.03
# --------------------------------------------------------------------------------------------------------------------------------------   
    
    metrics_dict = {'rmse': _rmse, 'mape': _mape, 'smape': _smape}
    # get climatology for the predicted months
    clim_pred = torch.Tensor(climatology[pred_months])
                    
    clim_per_timestep_scores = [{f'clim_{name}': metric(clim_pred.permute(0, 2, 1), twsa_true).mean(dim=0) for name, metric in metrics_dict.items()}][0]

    fcast_per_timestep_scores = [{f'{name}': metric(fcast, twsa_true).mean(dim=0) for name, metric in metrics_dict.items()}][0]
    fcast_basin_scores = [{f'{name}': metric(torch.Tensor(fcast_pred_basin_mean), torch.Tensor(true_basin_mean)).mean(dim=0) for name, metric in metrics_dict.items()}][0]
    fcast_basin_per_timestep_scores = [{f'{name}': metric(torch.Tensor(fcast_pred_basin_mean).permute(2, 0, 1), torch.Tensor(true_basin_mean).permute(2, 0, 1)).mean(dim=0) for name, metric in metrics_dict.items()}][0]

    gnn_per_timestep_scores = [{f'gnn_{name}': metric(twsa_pred, twsa_true).mean(dim=0) for name, metric in metrics_dict.items()}][0]


    # save scores to csv
    per_timestep_df = pd.DataFrame(fcast_per_timestep_scores, index=np.arange(6))
    per_timestep_df = per_timestep_df.assign(**clim_per_timestep_scores)
    per_timestep_df = per_timestep_df.assign(**gnn_per_timestep_scores)
    per_timestep_df.index.name = 'Timestep'
    per_timestep_df.to_csv(os.path.join(config["experiment"]["save_dir"],"fcastper_timestep_scores.csv"))
    basin_df = pd.DataFrame(fcast_basin_scores, index=basin_ids)
    basin_df.index.name = 'Basin ID'
    basin_df.to_csv(os.path.join(config["experiment"]["save_dir"],"fcast_basin_scores.csv"))
    basin_per_timestep_df = pd.DataFrame(fcast_basin_per_timestep_scores, index=np.arange(6))
    basin_per_timestep_df.index.name = 'Timestep'
    basin_per_timestep_df.to_csv(os.path.join(config["experiment"]["save_dir"],"fcast_basin_per_timestep_scores.csv"))
# --------------------------------------------------------------------------------------------------------------------------------------   
# calculate latitude weighted metrics - comparison with GRACE from  CSR RL06.03
# --------------------------------------------------------------------------------------------------------------------------------------   

    lat_weights = torch.cos(torch.deg2rad(torch.Tensor(twsa_sequence_data.data_train.land[0][:,1][antarctica_mask])))  # [N]
    # normalize so the weights sum to 1 (avoid changing overall loss scale)
    lat_weights = lat_weights / (lat_weights.sum() + 1e-12)
    lat_weighted_rmse = _lat_weighted_rmse(twsa_pred, twsa_true, lat_weights)
    lat_weighted_fcast_rmse = _lat_weighted_rmse(fcast, twsa_true, lat_weights)
    print(f'Latitude weighted RMSE for HiGNN: {lat_weighted_rmse.mean().item():.2f} cm')
    print(f'Latitude weighted RMSE for Li et al.: {lat_weighted_fcast_rmse.mean().item():.2f} cm')

# --------------------------------------------------------------------------------------------------------------------------------------   
# plot global mean 
# ------------------------------------------------------------- 
    
    plot_leadtimes_meantwsa(torch.nanmean(fcast, dim=1), 
                                 torch.mean(twsa_pred, dim=1), 
                                 torch.mean(twsa_true, dim=1),
                                 save_dir=os.path.join(config["experiment"]["save_dir"], f'GlobalMean_FCastvsGNN.png'))



# --------------------------------------------------------------------------------------------------------------------------------------   
# Plotting basin means - comparison
# ------------------------------------------------------------- 
    
    
    # Plotting basin means 

    start_year = timesteps[0]
    fig, axes = plt.subplots(4, 4, figsize=(10,10), sharex=True)
    fontsize = 12
    axes_flat = axes.flatten()
    
    for i, (name, id) in enumerate(TEST_BASIN_IDS.items()):
        ax = axes_flat[i]
        pred = pred_basin_mean[:,:,basin_ids == id] # shape (B, T)
        true = true_basin_mean[:,:,basin_ids == id]
        fcast = fcast_pred_basin_mean[:,:,basin_ids == id]
        plot_basintwsa_timesteps(name, id, pred, true, fcast, 
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
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('HiGNN-LSTM lead time (months)', fontsize=fontsize-2)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_ticks([1, 2, 3, 4, 5, 6])
    cbar.ax.tick_params(which='minor', length=0)
    cbar_ax2 = fig.add_axes([0.3, 0.05, 0.2, 0.02]) 
    cbar2 = fig.colorbar(sm2, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label('Li et al. lead time (months)', fontsize=fontsize-2)
    cbar2.ax.tick_params(labelsize=fontsize)
    cbar2.set_ticks([1, 2, 3, 4, 5, 6])
    cbar2.ax.tick_params(which='minor', length=0)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    
    plt.savefig(os.path.join(config["experiment"]["save_dir"], 
                                                      f'BasinPlots/FCastcomp_lessBasins_leadwise.png'), dpi=300)

         
   
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
    parser.add_argument(
        "--fcast_file_path", help="Path to the Li et al. forecast file (netcdf format)", default="data/fupeng_csr_fcast/combined.nc", type=str)

    args = parser.parse_args()

    evaluate(args.exp_path, gpu=args.gpu, dataset=args.dataset, FCast_file_path=args.fcast_file_path)


if __name__ == "__main__":
    start()
