
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import torch
from typing import Any, Dict
import xarray as xr

from datamodules.TWSA_heterograph_datamodule import TWSA_HeteroGraphDatamodule
from models.pl_hmmodel import HMModule
from utils.eval_metrics import (_corr, _lat_weighted_rmse, _rmse, get_basin_wide_mean_twsa,
                                get_climatology, get_months_output_sequences,
                                mask_coverage, recover_month_index, valid_mask)
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

# predicted months covered by the comparison (Li et al. initialisations 2019-10 to 2023-05)
COMPARISON_PERIOD = ('2019-11', '2023-11')


def select_comparison_window(first_target_times, output_seq_len=6, period=COMPARISON_PERIOD):
    """Mask of the samples whose predicted months all lie in `period`, and the Li et al.
    initialisation (the month before the first prediction) of each kept sample."""
    first = np.asarray(first_target_times).astype('datetime64[M]')
    keep = (first >= np.datetime64(period[0])) & (first + output_seq_len - 1 <= np.datetime64(period[1]))
    inits = [str(m - 1) for m in first[keep]]
    return keep, inits





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
    exp_dir_path = os.path.join(config["experiment"]["save_dir"], f"compare_to_fcast")
    os.makedirs(exp_dir_path, exist_ok=True)
    os.makedirs(os.path.join(exp_dir_path, "BasinPlots"), exist_ok=True)
    config["experiment"]["save_dir"] = exp_dir_path
    
    return config










def plot_basintwsa_timesteps(name, id, pred, true, fupeng,  ax, clim, fontsize=14):
    ''' pred shape [B, T], true shape [B, T], fupeng shape [B, T], clim shape [B, T] (training climatology
    of the predicted months, subtracted from all three for the anomaly correlation)
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
    
    pred_tensor = torch.tensor(pred, dtype=torch.float32).squeeze()  # [B, T]
    true_tensor = torch.tensor(true, dtype=torch.float32).squeeze()
    fupeng_tensor = torch.tensor(fupeng, dtype=torch.float32).squeeze()
    clim_tensor = torch.tensor(clim, dtype=torch.float32).squeeze()
    # _rmse reduces dim=1, so the windows go there: one RMSE per lead, then the mean
    # over the leads. _corr takes the windows on dim=0 and returns one r per lead --
    # correlating the 6 leads inside a window instead is blind to the window's level.
    rmse_pred = _rmse(pred_tensor.permute(1, 0), true_tensor.permute(1, 0)).mean()
    corr_pred = _corr(pred_tensor - clim_tensor, true_tensor - clim_tensor).mean()
    rmse_fup = _rmse(fupeng_tensor.permute(1, 0), true_tensor.permute(1, 0)).mean()
    corr_fup = _corr(fupeng_tensor - clim_tensor, true_tensor - clim_tensor).mean()

    # Add metrics text on the plot
    metrics_text = (f'$r_{{anom}}$={corr_pred:.2f}/{corr_fup:.2f}\n'
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
            ax[i].plot(x, f, label='Fupeng et al.', color = 'r',  linewidth=linewidth)
        else:
            ax[i].plot(x, t,  color='black', marker='o')
            ax[i].plot(x, p, color = 'b', linewidth=linewidth, )
            ax[i].plot(x, f, color = 'r',  linewidth=linewidth)
        ax[i].grid(True, which='both', axis='both', linestyle='--', color='gray', alpha=0.7)
        ax[i].set_title(f'{i+1} months lead')
        x+=1
    ax[0].legend(loc='upper right')
    plt.tight_layout()

    plt.savefig(save_dir, dpi=300)
    plt.close()



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

def get_predictions(twsa_sequence_data, trainer, config, experiment_path, dataset='test', gpu=0):
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
        # the checkpoint lives in the training experiment dir, not in the evaluation output dir
        model = build_model_from_checkpoint(experiment_path, config, gpu, twsa_sequence_data)

        # no trainer.test needed first: evaluate() assigns trainer.datamodule, which predict_step unnormalizes through
        predictions = trainer.predict(model, dataloader_pred) # predict usually selects val time period
        torch.save(predictions, os.path.join(config["experiment"]["save_dir"],"predictions.pt"))
    return predictions


def evaluate(experiment_path, gpu,
             FCast_file_path="data/Li_csr_fcast/combined.nc"):
    """Compares the model with the Li et al. forecast on the GRACE dataset.
    Args:
        experiment_path: path to the experiment directory
        gpu: gpu id to use
    """
    dataset = "grace"

    
    pl.seed_everything(42, workers=True)
    config = read_config(os.path.join(experiment_path,  'config.yaml'))
    # exchange devices to given gpu in config
    config['pl']['devices'] = [gpu]
    # the Li et al. forecast has 6 leads, the month bookkeeping below hardcodes 12 input
    # months, and get_months_output_sequences puts the first target right after the input
    assert config['datamodule']['input_seq_len'] == 12, "compare_models hardcodes input_seq_len=12"
    assert config['datamodule']['output_seq_len'] == 6, "compare_models hardcodes output_seq_len=6 (Li et al. has 6 leads)"
    assert config['datamodule']['delay'] == 0, "get_months_output_sequences assumes delay=0"

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
    predictions = get_predictions(twsa_sequence_data, trainer, config, experiment_path, dataset=dataset, gpu=gpu)

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

    ds_fcast = xr.open_dataset(FCast_file_path) # period between nov 2019 - nov 2023
    land_sea_mask = twsa_sequence_data.land_ocean_mask
    land_sea_mask = land_sea_mask[:6]
    
   
    # variables are named by initialisation date (YYYY-MM); pick the one matching each kept sample
    keep, inits = select_comparison_window(timesteps, output_seq_len)
    fcast = [ds_fcast[init].values[land_sea_mask].reshape(6,-1)[:, antarctica_mask] for init in inits]
    fcast = torch.Tensor(np.stack(fcast)).permute(0,2,1) # shape num_pred_seqs, nodes, seq_len


    # crop our prediction to the samples predicting Nov 2019 - Nov 2023, the Li et al. comparison period
    keep_t = torch.from_numpy(keep)
    twsa_pred = twsa_pred[keep_t]
    twsa_true = twsa_true[keep_t]
    timesteps = timesteps[keep]
    months_output_sequences = months_output_sequences[keep_t]  # else the climatology is off

    # One NaN policy for both tables: a cell is scored only where the target and
    # *both* models have data, so the gridded scores and the basin means are
    # computed over identical cells and the two models share one denominator.
    cell_mask = valid_mask(twsa_pred, fcast, twsa_true)
    twsa_pred = twsa_pred.masked_fill(~cell_mask, float('nan'))
    twsa_true = twsa_true.masked_fill(~cell_mask, float('nan'))
    fcast = fcast.masked_fill(~cell_mask, float('nan'))
    coverage = mask_coverage(cell_mask).mean(dim=0)  # [horizon]

    lat_weights = torch.cos(torch.deg2rad(torch.Tensor(twsa_sequence_data.data_train.land[0][:,1][antarctica_mask])))  # [N]
    # normalize so the weights sum to 1 (avoid changing overall loss scale)
    lat_weights = lat_weights / (lat_weights.sum() + 1e-12)

    pred_basin_mean, true_basin_mean, clim_basin_mean, pred_months, basin_ids = get_basin_wide_mean_twsa(twsa_pred, twsa_true, climatology, months_output_sequences, twsa_sequence_data, lat_weights=lat_weights)



    fcast_pred_basin_mean, _, _, _, _ = get_basin_wide_mean_twsa(fcast, twsa_true, climatology, months_output_sequences, twsa_sequence_data, lat_weights=lat_weights)


# --------------------------------------------------------------------------------------------------------------------------------------   
# calculate metrics - comparison with GRACE from  CSR RL06.03
# --------------------------------------------------------------------------------------------------------------------------------------   
    
    metrics_dict = {'rmse': _rmse}
    # get climatology for the predicted months
    clim_pred = torch.Tensor(climatology[pred_months])

    clim_per_timestep_scores = [{f'clim_{name}': metric(clim_pred.permute(0, 2, 1), twsa_true, mask=cell_mask).mean(dim=0) for name, metric in metrics_dict.items()}][0]
    clim_per_timestep_scores['clim_rmse_latw'] = _rmse(clim_pred.permute(0, 2, 1), twsa_true, mask=cell_mask, lat_weights=lat_weights).mean(dim=0)

    fcast_per_timestep_scores = [{f'{name}': metric(fcast, twsa_true, mask=cell_mask).mean(dim=0) for name, metric in metrics_dict.items()}][0]
    fcast_per_timestep_scores['coverage'] = coverage
    fcast_per_timestep_scores['rmse_latw'] = _rmse(fcast, twsa_true, mask=cell_mask, lat_weights=lat_weights).mean(dim=0)
    fcast_basin_scores = [{f'{name}': metric(torch.Tensor(fcast_pred_basin_mean), torch.Tensor(true_basin_mean)).mean(dim=0) for name, metric in metrics_dict.items()}][0]
    fcast_basin_per_timestep_scores = [{f'{name}': metric(torch.Tensor(fcast_pred_basin_mean).permute(2, 0, 1), torch.Tensor(true_basin_mean).permute(2, 0, 1)).mean(dim=0) for name, metric in metrics_dict.items()}][0]

    gnn_per_timestep_scores = [{f'gnn_{name}': metric(twsa_pred, twsa_true, mask=cell_mask).mean(dim=0) for name, metric in metrics_dict.items()}][0]
    gnn_per_timestep_scores['gnn_rmse_latw'] = _rmse(twsa_pred, twsa_true, mask=cell_mask, lat_weights=lat_weights).mean(dim=0)


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

    lat_weighted_rmse = _lat_weighted_rmse(twsa_pred, twsa_true, lat_weights, mask=cell_mask)
    lat_weighted_fcast_rmse = _lat_weighted_rmse(fcast, twsa_true, lat_weights, mask=cell_mask)
    print(f'Latitude weighted RMSE for HiGNN: {lat_weighted_rmse.mean().item():.2f} cm')
    print(f'Latitude weighted RMSE for Li et al.: {lat_weighted_fcast_rmse.mean().item():.2f} cm')
    # both are summed over the same cells, so both drop the same weight mass
    retained = (lat_weights[None, :, None] * cell_mask).sum(dim=1)
    print(f'Retained latitude weight mass: {retained.mean().item():.4f} of 1.0 '
          f'(cell coverage per lead: {[round(c, 4) for c in coverage.tolist()]})')

# --------------------------------------------------------------------------------------------------------------------------------------   
# plot global mean 
# ------------------------------------------------------------- 
    
    plot_leadtimes_meantwsa(torch.nanmean(fcast, dim=1),
                                 torch.nanmean(twsa_pred, dim=1),
                                 torch.nanmean(twsa_true, dim=1),
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
        fcast_basin = fcast_pred_basin_mean[:,:,basin_ids == id]
        plot_basintwsa_timesteps(name, id, pred, true, fcast_basin,
                                 ax=ax, clim=clim_basin_mean[:,:,basin_ids == id], fontsize=fontsize)

                                 
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
            '$r_{anom}$ / RMSE = HiGNN-LSTM / Li et al.',
            fontsize=fontsize-2,
            horizontalalignment='left',
            verticalalignment='center',        # center within the 0.02-height axes
            transform=text_ax.transAxes,       # coordinates relative to this axes
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
  

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
                                                      f'BasinPlots/FCastcomp_lessBasins_leadwise.png'), dpi=300,
                bbox_inches='tight')  # the r_anom note at the bottom right is wider than the figure

         
   
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
        "--fcast_file_path", help="Path to the Li et al. forecast file (netcdf format)", default="data/Li_csr_fcast/combined.nc", type=str)

    args = parser.parse_args()

    evaluate(args.exp_path, gpu=args.gpu, FCast_file_path=args.fcast_file_path)


if __name__ == "__main__":
    start()
