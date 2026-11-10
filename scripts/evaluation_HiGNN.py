import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import torch
from typing import Any, Dict


from datamodules.TWSA_heterograph_datamodule import TWSA_HeteroGraphDatamodule
from models.pl_hmmodel import HMModule
from utils.eval_metrics import (_corr, _rmse, get_basin_wide_mean_twsa, get_climatology,
                                get_months_output_sequences, gridded_score,
                                gridded_skill_score, load_grace_climatology, mask_coverage,
                                per_lead_skill_scores, recover_month_index, valid_mask)
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


def find_month_gap(timesteps):
    """Locate the widest gap in a monthly series of prediction dates.

    Returns (last index before the gap, first index after it), or (None, None) if
    the series is contiguous. Only the GRACE record has a gap; val and test do not.
    """
    dates = pd.to_datetime(timesteps)
    steps = np.diff(dates.year * 12 + dates.month)
    gaps = np.flatnonzero(steps > 1)
    if len(gaps) == 0:
        return None, None
    widest = int(gaps[np.argmax(steps[gaps])])
    return widest, widest + 1


def create_experiment_dir(config: Dict[str, Any], dataset: str="test") -> str:
    """Create experiment directory.

    Args:
        config: config file
        dataset: dataset name (test or grace)

    Returns:
        config with updated save_dir
    """
    exp_dir_path = os.path.join(config["experiment"]["save_dir"], f"eval_on_{dataset}")
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
        axes[2,i].set_xlabel(f"MAD: {abs(ape).nanmean():.2f} cm", fontsize=fontsize)
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

    # r over the windows for each lead time, then averaged over the leads -- the same
    # orientation as the Corrlead_deseas columns of basin_scores.csv. Correlating the 6 leads
    # within a window instead would score 6 points against 6 points. Anomaly correlation:
    # the basin climatology is removed from both, so the seasonal cycle adds no skill.
    corr = _corr(pred[:,:,0] - clim[:,:,0], true[:,:,0] - clim[:,:,0]).mean()
    ax.set_title(f'{name} ($r_{{anom}}$={corr:.2f})', fontsize=fontsize)

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
    # the best-val_rmse checkpoint is the only .ckpt in the experiment dir besides last.ckpt
    ckpt_files = [p for p in glob.glob(os.path.join(experiment_path, "*.ckpt"))
                  if os.path.basename(p) != "last.ckpt"]
    assert len(ckpt_files) == 1, f"expected one best-val_rmse ckpt in {experiment_path}, got {ckpt_files}"
    best_ckpt = ckpt_files[0]
    print(f"Loading best-val_rmse checkpoint: {best_ckpt}")
    #load model from checkpoint
    model = HMModule.load_from_checkpoint(best_ckpt, **config["model"],
                                          num_input_features=twsa_sequence_data.num_features,
                                          num_ocean_features=twsa_sequence_data.num_ocean_features,
                                            use_ocean_nodes=twsa_sequence_data.use_ocean_nodes,
                                            sequence_length=twsa_sequence_data.input_seq_len,
                                            output_seq_len=twsa_sequence_data.target_seq_len, 
                                            land_sea_mask=None,
                                            map_location=f'cuda:{gpu}')
   

    model.model_summary()
    return model

def get_predictions(twsa_sequence_data, trainer, config, experiment_path, dataset='test', gpu=0,
                    grace_climatology=None, save=True):
    if dataset == "test":
        dataloader_pred = twsa_sequence_data.predict_dataloader(period=twsa_sequence_data.train_val_test_periods['test'])
    elif dataset == "val":
        dataloader_pred = twsa_sequence_data.predict_dataloader(period=twsa_sequence_data.train_val_test_periods['val'])
    elif dataset == "train":
        dataloader_pred = twsa_sequence_data.predict_dataloader(period=twsa_sequence_data.train_val_test_periods['train'])

    elif dataset == "grace":
        dataloader_pred = twsa_sequence_data.GRACE_dataloader()
    elif dataset == "grace_graceclim":
        dataloader_pred = twsa_sequence_data.GRACE_dataloader(twsa_climatology=grace_climatology)

    try:
        predictions = torch.load(os.path.join(config["experiment"]["save_dir"],"predictions.pt"))
    except FileNotFoundError:
        predictions = None
    # a cache from an older dataset or time window has a different number of samples
    expected_rows = len(twsa_sequence_data.data_pred.samples) * twsa_sequence_data.data_pred.num_land_nodes()
    if predictions is not None and sum(batch[0].shape[0] for batch in predictions) != expected_rows:
        print('Cached predictions.pt does not match the prediction dataset, predicting again')
        predictions = None
    if predictions is None:
        # the checkpoint lives in the training experiment dir, not in the evaluation output dir
        model = build_model_from_checkpoint(experiment_path, config, gpu, twsa_sequence_data)

        predictions = trainer.predict(model, dataloader_pred) # predict usually selects val time period
        if save:
            torch.save(predictions, os.path.join(config["experiment"]["save_dir"],"predictions.pt"))
    return predictions

def evaluate(experiment_path, gpu, dataset="test", scores_only=False):
    """Evaluation Script.
    Args:
        experiment_path: path to the experiment directory
        gpu: gpu id to use
        dataset: dataset name (test, val, grace or grace_graceclim)
        scores_only: write only the score csvs, no predictions.pt and no figures
    """

    config = read_config(os.path.join(experiment_path,  'config.yaml'))
    pl.seed_everything(config["experiment"].get("seed", 42), workers=True)
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

    # grace_graceclim: C_grace replaces the training C in the model's in/outputs and in every score
    grace_climatology = load_grace_climatology() if dataset == 'grace_graceclim' else None
    climatology = get_climatology(twsa_sequence_data, config, climatology=grace_climatology)
     # get list of length num_batches, each element is a tuple of (y_hat, y_true, input_features)
    predictions = get_predictions(twsa_sequence_data, trainer, config, experiment_path, dataset=dataset, gpu=gpu,
                                  grace_climatology=grace_climatology, save=not scores_only)
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


    
    lat_weights = torch.cos(torch.deg2rad(torch.Tensor(twsa_sequence_data.data_pred.land[0][:,1][antarctica_mask])))  # [N], cos(lat) area weights
    pred_basin_mean, true_basin_mean, basin_mean_climatology, pred_months, basin_ids = get_basin_wide_mean_twsa(twsa_pred, twsa_true, climatology, months_output_sequences, twsa_sequence_data, lat_weights=lat_weights)



     # get climatology for the predicted months
    clim_pred = torch.Tensor(climatology[pred_months])
    
    metrics_dict = {'rmse': _rmse}

    # score the model and the climatology baseline over the same cells
    gridded_mask = valid_mask(twsa_pred, clim_pred.permute(0, 2, 1), twsa_true)
    per_timestep_scores = [{f'{name}': metric(twsa_pred, twsa_true, mask=gridded_mask).mean(dim=0) for name, metric in metrics_dict.items()}][0]
    per_timestep_scores['rmse_latw'] = _rmse(twsa_pred, twsa_true, mask=gridded_mask, lat_weights=lat_weights).mean(dim=0)
    per_timestep_scores['coverage'] = mask_coverage(gridded_mask).mean(dim=0)
    clim_per_timestep_scores = [{f'clim_{name}': metric(clim_pred.permute(0, 2, 1), twsa_true, mask=gridded_mask).mean(dim=0) for name, metric in metrics_dict.items()}][0]
    clim_per_timestep_scores['clim_rmse_latw'] = _rmse(clim_pred.permute(0, 2, 1), twsa_true, mask=gridded_mask, lat_weights=lat_weights).mean(dim=0)
    basin_scores = [{f'{name}': metric(torch.Tensor(pred_basin_mean), torch.Tensor(true_basin_mean)).mean(dim=0) for name, metric in metrics_dict.items()}][0]
    basin_per_timestep_scores = [{f'{name}': metric(torch.Tensor(pred_basin_mean).permute(2, 0, 1), torch.Tensor(true_basin_mean).permute(2, 0, 1)).mean(dim=0) for name, metric in metrics_dict.items()}][0]
    corr_basin_timesteps = [_corr(pred_basin_mean[:,:,i], true_basin_mean[:,:,i]) for i in range(true_basin_mean.shape[-1])]
    deseas_corr_basin_ts = [_corr(pred_basin_mean[:,:,i]-basin_mean_climatology[:,:,i], true_basin_mean[:,:,i]-basin_mean_climatology[:,:,i]) for i in range(true_basin_mean.shape[-1])]
    

    # save scores to csv
    per_timestep_df = pd.DataFrame(per_timestep_scores, index=np.arange(6))
    per_timestep_df = per_timestep_df.assign(**clim_per_timestep_scores)
    per_timestep_df = per_timestep_df.assign(**per_lead_skill_scores(
        twsa_pred, twsa_true, clim_pred.permute(0, 2, 1), lat_weights))
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
    if scores_only:
        return


# --------------------------------------------------------------------------------------------------------------
 # Plotting predictions and scores
 # --------------------------------------------------------------------------------------------------------------
    # plots for error drift per timestep
    from utils.utils import plot_rmse_broken_axis

    rmses = _rmse(twsa_pred, twsa_true)
    # break the axis only where the series really is discontinuous (the GRACE gap);
    # val and test are contiguous and get a single panel
    gap_start_id, gap_end_id = find_month_gap(timesteps)
    fig, axes = plot_rmse_broken_axis(
    dates        = timesteps,      # length N datetime array
    rmse         = rmses.transpose(1,0),      # shape (6, N)
    gap_start_id = gap_start_id,         # last index before the gap, or None
    gap_end_id   = gap_end_id,           # first index after the gap, or None
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
    
    # NSE, KGE and correlation scores
    from scores.continuous import nse, kge
    from scores.continuous.correlation import pearsonr

    # every score reduces over the windows at a fixed lead, so it describes each
    # node's series: reducing the 6 leads inside a window instead benchmarks that
    # window against its own mean. CorrDeseas is the anomaly correlation -- raw
    # TWSA r is carried by the annual cycle the climatology already reproduces.
    # NSEclim and CorrDeseas are the pair that isolate skill beyond seasonality:
    # NSEclim benchmarks against predicting the climatology instead of against the
    # pixel's test-period mean, and CorrDeseas correlates the anomalies.
    clim_gridded = clim_pred.permute(0, 2, 1)  # [windows, nodes, leads], as twsa_pred
    scores = [('NSE', gridded_score(nse, twsa_pred, twsa_true), 0),
              ('NSEclim', gridded_skill_score(twsa_pred, twsa_true, clim_gridded), 0),
              ('KGE', gridded_score(kge, twsa_pred, twsa_true), -0.41),
              ('Corr', gridded_score(pearsonr, twsa_pred, twsa_true), 0),
              ('CorrDeseas', gridded_score(pearsonr, twsa_pred, twsa_true,
                                           climatology=clim_gridded), 0)]

    # Plotting KGE, NSE scores and correlattion. Skill against lead time is the
    # point of a 6-step forecast, so each score keeps its leads next to the mean.
    for label, score, vmin in scores:
        os.makedirs(os.path.join(config["experiment"]["save_dir"], label), exist_ok=True)
        plot_NSE_or_KGE(score.mean(dim='lead'), twsa_sequence_data,
                        title=f'{label} (lead mean)', vmin=vmin,
                        save_dir=os.path.join(config["experiment"]["save_dir"],
                                              f'{label}/{label}mean.png'))
        for lead in range(score.sizes['lead']):
            plot_NSE_or_KGE(score.isel(lead=lead), twsa_sequence_data,
                            title=f'{label} (lead {lead + 1})', vmin=vmin,
                            save_dir=os.path.join(config["experiment"]["save_dir"],
                                                  f'{label}/{label}_lead{lead + 1}.png'))
    # -----------------------------------------------------------
     # Plotting basin means 
    # ------------------------------------------------------------ 
    if dataset in ('grace', 'grace_graceclim'):
        # first sample predicting Nov 2019, the first predicted month after the GRACE/GRACE-FO gap
        start_month_idx = int(np.searchsorted(timesteps.astype('datetime64[M]'), np.datetime64('2019-11')))
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

    # # -----------------------------------------------------------
    # # plot differnet time period
    # # ------------------------------------------------------------
    # end_month_idx = 17
    # start_year_month = timesteps[0]
    # start_year_month = twsa_sequence_data.data_pred.ds_timeslice.time.values[0]+ pd.DateOffset(years=1)
    # fig, axes = plt.subplots(4, 4, figsize=(15,15), sharex=True)
    # axes_flat = axes.flatten()
    
    # for i, (name, id) in enumerate(TEST_BASIN_IDS.items()):
    #     if i == 16:
    #         break
    #     ax = axes_flat[i]
    #     pred = pred_basin_mean[:end_month_idx,:,basin_ids == id] # shape (B, T)
    #     true = true_basin_mean[:end_month_idx,:,basin_ids == id]
    #     clim = basin_mean_climatology[:end_month_idx,:,basin_ids == id]
    #     plot_basintwsa_timesteps(name, id, pred, true, clim, 
    #                               ax=ax)
    #     # start x labels from +1 instead of 0 because of 12 months input 
        
    #     if i > 11:
           
    #         x_labels = [(start_year_month + pd.DateOffset(years=i)).strftime('%b %Y') for i in range(((len(pred)+6)//12+1))]
    #         ax.set_xticks(list(np.arange(0,len(pred)+6+1, 12)))
    #         ax.set_xticklabels(x_labels, rotation=90, fontsize=12)
    #     else: 
    #         ax.set_xticklabels([])
    #     if i in [0,4,8,12]:
    #         ax.set_ylabel('TWSA [cm]', fontsize=12)
    #     ax.tick_params(axis='both', which='major', labelsize=14)
    # # ax.legend(loc='best', fontsize=12)
    # handles, labels = ax.get_legend_handles_labels()
    # legend_ax = fig.add_axes([0.60, 0.05, 0.2, 0.02]) 
    # legend_ax.axis("off")
    # legend_ax.legend(handles, labels,
    #         loc="right",
    #         fontsize=14)
    
    # # Add colorbar below all plots
    # lead_time_cmap = plt.cm.winter
    # sm = plt.cm.ScalarMappable(cmap=lead_time_cmap, norm=plt.Normalize(vmin=1, vmax=6))
    # sm.set_array([])
    # cbar_ax = fig.add_axes([0.15, 0.05, 0.4, 0.02])  # [left, bottom, width, height]
    # cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    # cbar.set_label('HiGNN-LSTM lead time (months)', fontsize=14)
    # cbar.ax.tick_params(labelsize=14)
    # cbar.set_ticks([1, 2, 3, 4, 5, 6])
    # plt.tight_layout(rect=[0, 0.1, 1, 1])
    # plt.savefig(os.path.join(config["experiment"]["save_dir"], 
    #                                                   f'BasinPlots/All_BasinsSeq0.png'), dpi=300)



 
   
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
        "--dataset", help="Dataset to evaluate on (test, val, grace, or grace_graceclim)", default="test", type=str)
    parser.add_argument("--scores_only", action="store_true",
                        help="Write only the score csvs, no predictions.pt and no figures")

    args = parser.parse_args()

    evaluate(args.exp_path, gpu=args.gpu, dataset=args.dataset, scores_only=args.scores_only)


if __name__ == "__main__":
    start()
