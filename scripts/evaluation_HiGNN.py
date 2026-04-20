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
                    "Murray": 5030073410, # AUS
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
    exp_dir_path = os.path.join(f"eval_on_{dataset}", config["experiment"]["save_dir"])
    os.makedirs(exp_dir_path, exist_ok=True)
    os.makedirs(os.path.join(exp_dir_path, "KGE"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir_path, "BasinPlots"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir_path, "Predictions"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir_path, "NSE"), exist_ok=True)
    config["experiment"]["save_dir"] = exp_dir_path
    
    return config
                 

def plot_prediction(y_hat, y_true, datamodule, antarctica=None, start_date=None,
                    save_dir=None, title='TWSA Prediction (Test Set)'):
    """y_hat, y_true: [N, T]"""
    lon, lat = datamodule.data_pred.land[0][:,0], datamodule.data_pred.land[0][:,1]

    if antarctica is not None:
        antarctica_mask =  (~datamodule.get_antarctica_landmask())
        lon = lon[antarctica_mask]
        lat = lat[antarctica_mask]

    vmax = 40
    f, axes = plt.subplots(3,6,figsize=(8,5))
    fontsize = 10
    for i in range(y_hat.shape[1]):
        p1 = axes[0,i].scatter(lon, lat, c=y_hat[:,i], cmap='coolwarm',marker='.', s=1,vmin=-vmax, vmax=vmax)
        p2 = axes[1,i].scatter(lon, lat, c=y_true[:,i], cmap='coolwarm',marker='.', s=1,vmin=-vmax, vmax=vmax)
        ape = (y_hat[:,i]-y_true[:,i])
        p3 = axes[2,i].scatter(lon, lat, c=ape, cmap='RdBu',s=1, marker='.', vmin=-10, vmax=10)
        # suptitle with mean APE
        axes[2,i].set_xlabel(f"MAD: {abs(ape).mean():.2f} cm", fontsize=fontsize)
        axes[0,i].set_title((start_date + pd.DateOffset(months=i)).strftime('%b %Y'), fontsize=fontsize+1)
        
        # remove axes tick labels
        format_global_plot(axes[0,i], noticks_x=True, noticks_y=True)
        format_global_plot(axes[1,i], noticks_x=True, noticks_y=True)
        format_global_plot(axes[2,i], noticks_x=True, noticks_y=True)

    cbar_ticks = [-vmax,0,vmax]

    cbar1 = plt.colorbar(p1, ax=axes[0,5], extend='both')
    cbar1.set_label('TWSA [cm]', fontsize=fontsize)
    cbar1.set_ticks(cbar_ticks)
    
    cbar2 = plt.colorbar(p2, ax=axes[1,5], extend='both')
    cbar2.set_label('TWSA [cm]', fontsize=fontsize)
    cbar2.set_ticks(cbar_ticks)
    cbar3 = plt.colorbar(p3, ax=axes[2,5], extend='both')
    cbar3.set_label('Difference [cm]', fontsize=fontsize)
    cbar3.set_ticks([-10, 10])

    axes[0,0].set_ylabel('Prediction', fontsize=fontsize)
    axes[1,0].set_ylabel('Target', fontsize=fontsize)
    axes[2,0].set_ylabel('Pred-Target', fontsize=fontsize)
    plt.tight_layout()
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
    return torch.mean(torch.abs((y_true - y_hat) / (y_true + epsilon)), dim=1) * 100  # in percentage

def _rmse(y_hat, y_true):
    """Calculate Root Mean Squared Error (RMSE) between two tensors."""
    return torch.sqrt(torch.mean((y_true - y_hat) ** 2, dim=1))  # RMSE

def _smape(y_hat, y_true):
    """Calculate Symmetric Mean Absolute Percentage Error (SMAPE) between two tensors."""
    # Avoid division by zero
    epsilon = 1e-8
    return torch.mean(torch.abs(y_true - y_hat) / (torch.abs(y_true) + torch.abs(y_hat) + epsilon), dim=1) * 100  # in percentage

def _corr(y_hat, y_true):
    pearson = torchmetrics.PearsonCorrCoef(num_outputs=6)
    return pearson(torch.Tensor(y_hat), torch.Tensor(y_true))

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

def plot_basintwsa_timesteps(name, id, pred, true, clim,  ax, fontsize=14):
    ''' pred shape [B, T], true shape [B, T], clim shape [B, T] 
    Colors sequences with gradient from light to dark blue based on lead time (1-6)
    '''
    x_base = np.arange(6)
    linewidth = 1.0
    
    # Colormap for lead time gradient
    lead_time_cmap = plt.cm.winter

    for seq_idx, (p, t, c) in enumerate(zip(pred, true, clim)):
        x = x_base + seq_idx  # Offset for each sequence
        
        if seq_idx == 0:
            ax.plot(x, t, label='Target TWSA', color='black', marker='.', linewidth=1.5)

        else: 
            ax.plot(x, t, color='black', marker='.', linewidth=1.5)
        
        # Plot prediction with gradient color based on lead time
        for j in range(len(p) - 1):
            # Color based on lead time position (1-6 normalized to 0-1)
            color = lead_time_cmap((j + 1) / 6.0)
            # Plot segment from lead time j to j+1
            ax.plot(x[j:j+2], p[j:j+2], color=color, linewidth=linewidth)
        
        # Plot last point
        ax.plot(x[-1], p[-1], color=lead_time_cmap(1.), linewidth=linewidth)
        _corr = torchmetrics.PearsonCorrCoef(num_outputs=pred.shape[0])
    corr = _corr(torch.Tensor(pred).squeeze().permute(1,0), torch.Tensor(true).squeeze().permute(1,0)).mean()
    ax.set_title(f'{name} (r={corr:.2f})', fontsize=fontsize)

    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    
    ax.grid(True, which='major', axis='both', linestyle='--', color='gray', alpha=0.7)
   
    return ax

def plot_basintwsa_per_lead(name, id, pred, true, seasonal, ax=None):
    ''' pred shape [B, T], true shape [B, T], clim shape [B, T] '''
    x = np.arange(pred.shape[0])
    pred = np.transpose(pred[:,:,0])

    true = np.transpose(true[:,:,0])
    seasonal = np.transpose(seasonal[:,:,0])
    linewidth = 1

    cmapgnn = plt.cm.Reds

    colors_gnn = [cmapgnn(i / 10) for i in range(10)]
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,8))
    
    ax.plot(x, seasonal[0], label='Seasonal Trend', linestyle='dotted', color='gray', linewidth=linewidth*2)

    for i, (p, t) in enumerate(zip(pred, true)):
        if i ==0:
            ax.plot(x, t, label='GRACE-FO CSR RL06.3 ', color='black', marker='o')
            ax.plot(x, p, label='HiGNN', color = colors_gnn[i+4],  linewidth=linewidth)

        else:
            i = i % 6
            ax.plot(x, t,  color='black', marker='o')
            ax.plot(x, p, color = colors_gnn[i+4], linewidth=linewidth, )

        x+=1

    ax.set_title(f'{name} ({id})', fontsize=13, fontweight='bold')

    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, which='both', axis='both', linestyle='--', color='gray', alpha=0.7)

    return ax


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
    
    pl.seed_everything(42, workers=True)
    config = read_config(os.path.join(experiment_path,  'config.yaml'))
    # exchange devices to given gpu in config
    config['pl']['devices'] = [gpu]

    config = create_experiment_dir(config, dataset=dataset)

    # generate trainer
    trainer = generate_trainer(config)

    twsa_sequence_data = TWSA_HeteroGraphDatamodule(**config['datamodule'])
    twsa_sequence_data.setup(stage='fit') #initializes data_train and data val from TWSA-rec data
    output_seq_len = config['datamodule']['output_seq_len']
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
        
    
   
    time_period = twsa_sequence_data.data_pred.time_period
    print('Prediction time period:' , time_period)
    num_nodes = twsa_sequence_data.data_pred.num_land_nodes() 
    # get mask to mask antarctica out in the evaluation
    antarctica_mask =  (~twsa_sequence_data.get_antarctica_landmask())
    climatology = climatology[:,antarctica_mask]  # [12, N]

    
    pred_basin_mean, true_basin_mean, basin_mean_climatology, pred_months, basin_ids = get_basin_wide_mean_twsa(twsa_pred, twsa_true, climatology, months_output_sequences, twsa_sequence_data)



     # get climatology for the predicted months
    clim_pred = torch.Tensor(climatology[pred_months])
    
    metrics_dict = {'rmse': _rmse, 'mape': _mape, 'smape': _smape}
                    
    per_timestep_scores = [{f'{name}': metric(twsa_pred, twsa_true).mean(dim=0) for name, metric in metrics_dict.items()}][0]
    clim_per_timestep_scores = [{f'clim_{name}': metric(clim_pred.permute(0, 2, 1), twsa_true).mean(dim=0) for name, metric in metrics_dict.items()}][0]
    basin_scores = [{f'{name}': metric(torch.Tensor(pred_basin_mean), torch.Tensor(true_basin_mean)).mean(dim=0) for name, metric in metrics_dict.items()}][0]
    basin_per_timestep_scores = [{f'{name}': metric(torch.Tensor(pred_basin_mean).permute(2, 0, 1), torch.Tensor(true_basin_mean).permute(2, 0, 1)).mean(dim=0) for name, metric in metrics_dict.items()}][0]
    corr_basin_timesteps = [_corr(pred_basin_mean[:,:,i], true_basin_mean[:,:,i]) for i in range(true_basin_mean.shape[-1])]
    deseas_corr_basin_ts = [_corr(pred_basin_mean[:,:,i]-basin_mean_climatology[:,:,i], true_basin_mean[:,:,i]-basin_mean_climatology[:,:,i]) for i in range(true_basin_mean.shape[-1])]
    

    # save scores to csv
    per_timestep_df = pd.DataFrame(per_timestep_scores, index=np.arange(6))
    per_timestep_df = per_timestep_df.assign(**clim_per_timestep_scores)
    per_timestep_df.index.name = 'Timestep'
    per_timestep_df.to_csv(os.path.join(config["experiment"]["save_dir"],"per_timestep_scores.csv"))
    basin_df = pd.DataFrame(basin_scores, index=basin_ids)
    # add corr_basin_timesteps to df
    lead_corrs = pd.DataFrame(corr_basin_timesteps, index=basin_ids, columns=[f'Corrlead_{i+1}' for i in range(6)])
    lead_corrs_deseas = pd.DataFrame(deseas_corr_basin_ts, index=basin_ids, columns=[f'Corrlead{i+1}_deseas' for i in range(6)])
    basin_df = pd.concat([basin_df, lead_corrs.astype('float64'), lead_corrs_deseas.astype('float64')], axis=1)
    basin_df.index.name = 'Basin ID'
    basin_df.to_csv(os.path.join(config["experiment"]["save_dir"],"basin_scores.csv"))
    basin_per_timestep_df = pd.DataFrame(basin_per_timestep_scores, index=np.arange(6))
    basin_per_timestep_df.index.name = 'Timestep'
    basin_per_timestep_df.to_csv(os.path.join(config["experiment"]["save_dir"],"basin_per_timestep_scores.csv"))


# --------------------------------------------------------------------------------------------------------------
 # Plotting predictions and scores
 # --------------------------------------------------------------------------------------------------------------
    # plots for error drift per timestep
    from utils.utils import plot_rmse_broken_axis

    rmses = _rmse(twsa_pred, twsa_true)
    fig, axes = plot_rmse_broken_axis(
    dates        = timesteps,      # length N datetime array
    rmse         = rmses.transpose(1,0),      # shape (6, N)
    gap_start_id = 16,                   # index of Oct 2010
    gap_end_id   = 17,                   # index of Nov 2019
    lead_labels  = [f"Lead {i+1}" for i in range(6)],
    cmap_name    = "Blues_r",            # swap to "Reds_r" etc. if needed
    savepath     = os.path.join(config["experiment"]["save_dir"],"rmse_monthly.png"),
)
    plt.close()

    # --------------------------------------------------------------------------------------------------------------
    # boxplots for corrs
    # --------------------------------------------------------------------------------------------------------------

    ax = basin_df.astype('float64').boxplot(column=[f'Corrlead{i+1}_deseas' for i in range(6)])
    ax.set_xticklabels([f'Lead {i+1}' for i in range(6)])
    plt.savefig(os.path.join(config["experiment"]["save_dir"],f"BasinCorrsDeseas"))
    plt.close()
    ax=basin_df.astype('float64').boxplot(column=[f'Corrlead_{i+1}' for i in range(6)])
    ax.set_xticklabels([f'Lead {i+1}' for i in range(6)])
    plt.savefig(os.path.join(config["experiment"]["save_dir"], f"BasinCorrs"))
    plt.close()
   
    # Plotting predictions 
    #for i in range(twsa_pred.shape[0]):
    if True:
        for i in range(twsa_pred.shape[0]):
            plot_prediction(twsa_pred[i], twsa_true[i], twsa_sequence_data, antarctica=True, 
                        start_date=timesteps[i], #pd.to_datetime(time_period[0])+ pd.DateOffset(months=6-1),
                                save_dir=os.path.join(config["experiment"]["save_dir"], 
                                                      f"Predictions/Seq{i}"), 
                                                      title=None)
            # exit the loop after plotting the first sequence s
            break
    # for i, (pred, true) in enumerate(zip(twsa_pred, twsa_true)):
    #     plot_prediction(pred, true, twsa_sequence_data, antarctica=True, 
    #                     start_date=pd.to_datetime(time_period[0])+ pd.DateOffset(years=1) + pd.DateOffset(months=i),
    #                             save_dir=os.path.join(config["experiment"]["save_dir"], 
    #                                                   f"Predictions/Seq_{i}"), 
    #                                                   title=None)
    
    # NSE, KGE scores
    from scores.continuous import nse, kge
    from scores.continuous.correlation import pearsonr
    
    # calculate NSE accros all sequences
    arr_pred = xr.DataArray(twsa_pred, dims=['batch', 'nodes', 'time'], coords={'batch': np.arange(twsa_pred.shape[0]), 
                                                                            'nodes': np.arange(twsa_pred.shape[1]),
                                                                            'time': np.arange(twsa_pred.shape[2])})
    arr_true = xr.DataArray(twsa_true, dims=['batch','nodes', 'time'], coords={'batch': np.arange(twsa_true.shape[0]), 
                                                                               'nodes': np.arange(twsa_true.shape[1]), 
                                                                               'time': np.arange(twsa_true.shape[2])})
    nse_score = nse(arr_pred, arr_true, reduce_dims=['time'])
    kge_score = kge(arr_pred, arr_true, reduce_dims=['time'])
    corr_score = pearsonr(arr_pred, arr_true, reduce_dims=['time'])
    
    # Plotting KGE, NSE scores and correlattion
    plot_NSE_or_KGE(nse_score.mean(dim='batch'), twsa_sequence_data, title='NSE', vmin=0,
                    save_dir=os.path.join(config["experiment"]["save_dir"], f'NSE/NSEmean.png'))
    plot_NSE_or_KGE(kge_score.mean(dim='batch'), twsa_sequence_data, title='KGE', vmin=-0.41,
                    save_dir=os.path.join(config["experiment"]["save_dir"], f'KGE/KGEmean.png'))
    plot_NSE_or_KGE(corr_score.mean(dim='batch'), twsa_sequence_data, title='Correlation', vmin=0,
                    save_dir=os.path.join(config["experiment"]["save_dir"], f'Corrmean.png'))
    # -----------------------------------------------------------
     # Plotting basin means 
    # ------------------------------------------------------------ 
    if dataset == 'grace':
        start_month_idx = 17
    else: start_month_idx = 0
    start_year_month = timesteps[start_month_idx]
    fig, axes = plt.subplots(4, 4, figsize=(8,8), sharex=True)
    fontsize = 10
    axes_flat = axes.flatten()
    
    for i, (name, id) in enumerate(TEST_BASIN_IDS.items()):
        if i == 16:
            break
        ax = axes_flat[i]
        pred = pred_basin_mean[start_month_idx:,:,basin_ids == id] # shape (B, T)
        true = true_basin_mean[start_month_idx:,:,basin_ids == id]
        clim = basin_mean_climatology[start_month_idx:,:,basin_ids == id]
        plot_basintwsa_timesteps(name, id, pred, true, clim, 
                                  ax=ax, fontsize=fontsize)
        # start x labels from +1 instead of 0 because of 12 months input 
        
        if i > 11:
           
            x_labels = [(start_year_month + pd.DateOffset(years=i)).strftime('%b %Y') for i in range(((len(pred)+6)//12+1))]
            ax.set_xticks(list(np.arange(0,len(pred)+6+1, 12)))
            ax.set_xticklabels(x_labels, rotation=90, fontsize=fontsize)
        else: 
            ax.set_xticklabels([])
        if i in [0,4,8,12]:
            ax.set_ylabel('TWSA [cm]', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

    handles, labels = ax.get_legend_handles_labels()
    legend_ax = fig.add_axes([0.6, 0.05, 0.2, 0.02]) 
    legend_ax.axis("off")
    legend_ax.legend(handles, labels,
            loc="right",
            fontsize=fontsize)
    
    # Add colorbar below all plots
    lead_time_cmap = plt.cm.winter 
    sm = plt.cm.ScalarMappable(cmap=lead_time_cmap, norm=plt.Normalize(vmin=1, vmax=6))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.05, 0.4, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('HiGNN-LSTM lead time (months)', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_ticks([1, 2, 3, 4, 5, 6])
    cbar.set_ticklabels(['1', '2', '3', '4', '5', '6'])
    cbar.ax.tick_params(which='minor', length=0)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(os.path.join(config["experiment"]["save_dir"], 
                                                      f'BasinPlots/All_Basins.png'), dpi=300)

    # -----------------------------------------------------------
    # plot differnet time period
    # ------------------------------------------------------------
    end_month_idx = 17
    start_year_month = timesteps[0]
    start_year_month = twsa_sequence_data.data_pred.ds_timeslice.time.values[0]+ pd.DateOffset(years=1)
    fig, axes = plt.subplots(4, 4, figsize=(15,15), sharex=True)
    axes_flat = axes.flatten()
    
    for i, (name, id) in enumerate(TEST_BASIN_IDS.items()):
        if i == 16:
            break
        ax = axes_flat[i]
        pred = pred_basin_mean[:end_month_idx,:,basin_ids == id] # shape (B, T)
        true = true_basin_mean[:end_month_idx,:,basin_ids == id]
        clim = basin_mean_climatology[:end_month_idx,:,basin_ids == id]
        plot_basintwsa_timesteps(name, id, pred, true, clim, 
                                  ax=ax)
        # start x labels from +1 instead of 0 because of 12 months input 
        
        if i > 11:
           
            x_labels = [(start_year_month + pd.DateOffset(years=i)).strftime('%b %Y') for i in range(((len(pred)+6)//12+1))]
            ax.set_xticks(list(np.arange(0,len(pred)+6+1, 12)))
            ax.set_xticklabels(x_labels, rotation=90, fontsize=12)
        else: 
            ax.set_xticklabels([])
        if i in [0,4,8,12]:
            ax.set_ylabel('TWSA [cm]', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=14)
    # ax.legend(loc='best', fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    legend_ax = fig.add_axes([0.60, 0.05, 0.2, 0.02]) 
    legend_ax.axis("off")
    legend_ax.legend(handles, labels,
            loc="right",
            fontsize=14)
    
    # Add colorbar below all plots
    lead_time_cmap = plt.cm.winter
    sm = plt.cm.ScalarMappable(cmap=lead_time_cmap, norm=plt.Normalize(vmin=1, vmax=6))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.05, 0.4, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('HiGNN-LSTM lead time (months)', fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_ticks([1, 2, 3, 4, 5, 6])
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(os.path.join(config["experiment"]["save_dir"], 
                                                      f'BasinPlots/All_BasinsSeq0.png'), dpi=300)



 
   
    # --------------------------------------------------------------------
    # save config
    # --------------------------------------------------------------------
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
