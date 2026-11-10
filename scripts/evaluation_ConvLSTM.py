
import argparse
import glob
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import pytorch_lightning as pl

import torch
import torchmetrics
from typing import Any, Dict


from datamodules.TWSA_datamodule2D import TWSA_Datamodule2D

from models.pl_convlstm import ConvLSTMModule
from utils.eval_metrics import _rmse, load_grace_climatology, nse, per_lead_skill_scores, recover_month_index
from utils.utils import read_config, save_config


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

        )


def create_experiment_dir(config: Dict[str, Any], experiment_path, dataset: str="test") -> str:
    """Create experiment directory.

    Args:
        config: config file
        dataset: dataset name (test, val or grace)

    Returns:
        config with updated save_dir
    """
    exp_dir_path = os.path.join(experiment_path, f"eval_on_{dataset}")
    os.makedirs(exp_dir_path, exist_ok=True)
    config["experiment"]["save_dir"] = exp_dir_path
    
    return config

def plot_prediction(y_hat, y_true, datamodule, antarctica=None, save_dir=None, title='TWSA Prediction (Test Set)'):

    vmax = 100
    f, ( ax1, ax2, ax3) = plt.subplots(3,1,figsize=(5,10))
    p1 = ax1.imshow(y_hat, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    p2 = ax2.imshow(y_true, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    ape = abs(y_hat-y_true)/(abs(y_true)+ 1e-5) * 100
    p3 = ax3.imshow(ape, cmap='Reds', vmin=0, vmax=100)

    f.suptitle(title)
    format_global_plot(ax1, noticks_x=True, noticks_y=True)
    format_global_plot(ax2, noticks_x=True, noticks_y=True)
    format_global_plot(ax3, noticks_x=True, noticks_y=True)
    
    plt.tight_layout()
    cbar1 = plt.colorbar(p1, ax=ax1)
    cbar1.set_label('TWSA [cm]', fontsize=12)
    cbar1.set_ticks([-80,-40, 0, 40, 80])
    cbar2 = plt.colorbar(p2, ax=ax2)
    cbar2.set_label('TWSA [cm]', fontsize=12)
    cbar2.set_ticks([-80,-40, 0, 40, 80])
    cbar3 = plt.colorbar(p3, ax=ax3, extend='max')
    cbar3.set_label('Difference [%]', fontsize=12)
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
    land_mask = (datamodule.get_landmask_without_antarctica() & datamodule.data_train.land_ocean_mask[0])
    f, ax0 = plt.subplots(1,1, figsize=(5,4))
    # -0.41 everything below is worse than a mean prediction for each time step
    cmap = plt.cm.Reds.reversed()
    if vmin is not None:
        cmap.set_under('blue')
    p0 = ax0.imshow(nse, vmin=vmin, vmax=1,  cmap=cmap)
    # format_global_plot(ax0)
    plt.colorbar(p0, ax=ax0, extend='min')
    plt.figtext(0.5, 0.00, f'Mean {title}: {nse.mean().item():.2f}', ha='center', fontsize=10)

    ax0.set_title(title) 
    plt.tight_layout()

    plt.savefig(save_dir, dpi=300)
    plt.close()

def plot_KGE(kge, datamodule, save_dir=None):
    land_mask = (datamodule.get_landmask_without_antarctica() & datamodule.data_train.land_ocean_mask[0])
    f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(20,10))
    # -0.41 everything below is worse than a mean prediction for each time step
    p0 = ax0.imshow(kge[0],   vmin=-0.41)
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
    
    plt.text(0.5, -0.3, 'Mean KGE: {:.2f} '.format(np.nanmean(kge[0][land_mask])),  horizontalalignment='center',
     verticalalignment='center', transform=ax3.transAxes)
    
    plt.tight_layout()

    plt.savefig(save_dir)
    plt.close()

def plot_ssim(y_pred, y_true, datamodule, index, antarctica_mask=None, climatology=None, save_dir=None, f=None, ax=None):
    """Calculate the Structural Similarity Index (SSIM) between two tensors.
    y_pred: predicted values, shape [T, N]
    y_true: true values, shape [T, N]
    datamodule: datamodule containing the land_ocean_mask"""


    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(reduction='none')
    y_pred = y_pred.unsqueeze(1)
    y_true = y_true.unsqueeze(1)
    # compute SSIM for the current time step
    ssim = ssim_metric(y_pred, y_true)
    ssim_zeros = ssim_metric(torch.zeros(y_true.shape), y_true)
    if climatology is not None:
        ssim_clim = ssim_metric(torch.Tensor(climatology).unsqueeze(1), y_true)
    else: ssim_clim = None

    
    f, ax = plt.subplots(1,1, figsize=(5,3))
    # plot the SSIM values
    ax.plot(ssim.cpu().numpy(), label='Forecast vs. Ground Truth', marker='o')
    if climatology is not None:
        ax.plot(ssim_clim.cpu().numpy(), label='Climatology vs. Ground Truth', marker='o', color='orange')
    format_ssim_plot(ax, ssim_zeros, index)
    if save_dir is not None:
        plt.savefig(save_dir)
        plt.close()
    return f, ax, ssim, ssim_zeros, ssim_clim

def format_ssim_plot(ax, ssim_zeros, index):
    ax.plot(ssim_zeros.cpu().numpy(), label='Baseline SSIM (Zeros vs. Ground Truth)', marker='o', color='black', linestyle='--')
    ax.text(0, 1.01, 'Optimal Value', color='gray', fontsize=10, ha='left', va='bottom')
    ax.axhline(y=1, color='gray', linestyle='--')
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('SSIM')
    ax.set_title(f'SSIM over Timesteps of forecast - {index}')
    ax.set_ylim(-0.05, 1.07)
    ax.set_yticks(np.linspace(0,1.0,6))
    ax.legend()
    plt.tight_layout()
    return ax

def plot_explained_variance(y_hat, y_true, datamodule, save_dir=None):

    # calculate mean without antarctica as it is already removed in the predictions
    antarctica_mask = datamodule.get_landmask_without_antarctica()
    land_mask = datamodule.data_train.land_ocean_mask[0]
    mask = land_mask & antarctica_mask

    #lon, lat = datamodule.data_pred.land[0][:,0], datamodule.data_pred.land[0][:,1]
    
    # flip the cmap colors and add color for below 0 values
    cmap = plt.cm.Reds.reversed()
    cmap.set_under('blue')
    exp_var = torchmetrics.ExplainedVariance(multioutput='raw_values')(y_hat, y_true)
    f, ax = plt.subplots(1,1)
    p=ax.imshow(exp_var, cmap=cmap,  vmin=0)
    # p=ax.scatter(lon[antarctica_mask], lat[antarctica_mask], c=exp_var[antarctica_mask], cmap=cmap, s=1, vmin=0)
    ax.set_title(f'Explained Variance per Pixel')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.tight_layout()
    plt.colorbar(p, ax=ax, extend='min')
    # add caption with the mean explained variance 
    
    plt.figtext(0.5, 0.00, f'Mean Explained Variance: {exp_var[mask].mean().item():.2f}', ha='center', fontsize=12)
    if save_dir is not None:
        plt.savefig(save_dir)
        plt.close()

def calculate_per_timestep_scores(y_hat, y_true, climatology, scores_dict, metrics_dict, lsm=None):
    """Calculate per timestep scores for the given predictions and ground truth."""
    
    y_hat_deseas = y_hat - climatology
    y_true_deseas = y_true - climatology
    if lsm is not None:
        # mask out the antarctica pixels
        
        y_hat = y_hat[:, lsm]
        y_true = y_true[:, lsm]
        y_hat_deseas = y_hat_deseas[:, lsm]
        y_true_deseas = y_true_deseas[:, lsm]
        climatology = climatology[:, lsm]

    y_hat = y_hat.reshape(y_hat.shape[0], -1) # [seq_len, HxW]
    y_true = y_true.reshape(y_true.shape[0], -1)
    y_hat_deseas = y_hat_deseas.reshape(y_hat_deseas.shape[0], -1)
    y_true_deseas = y_true_deseas.reshape(y_true_deseas.shape[0], -1)
    climatology = climatology.reshape(climatology.shape[0], -1)
    
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


def evaluate(experiment_path, gpu, dataset="test"):
    """Evaluation Script."""
    # torch.set_float32_matmul_precision("medium")
    
    pl.seed_everything(42, workers=True)
    config = read_config(os.path.join(experiment_path,  'config.yaml'))
    # exchange devices to given gpu in config
    config['pl']['devices'] = [gpu]

    config = create_experiment_dir(config, experiment_path, dataset=dataset)

    # generate trainer
    trainer = generate_trainer(config)

    twsa_sequence_data = TWSA_Datamodule2D(**config['datamodule'])
    twsa_sequence_data.setup(stage='fit')
    output_seq_len = config['datamodule']['output_seq_len']
    trainer.datamodule = twsa_sequence_data # assign datamodule to trainer for prediction

    

    # the best-val_rmse checkpoint is the only .ckpt in the experiment dir besides last.ckpt
    ckpt_files = [p for p in glob.glob(os.path.join(experiment_path, "**", "*.ckpt"), recursive=True)
                  if os.path.basename(p) != "last.ckpt"]
    assert len(ckpt_files) == 1, f"expected one best-val_rmse ckpt in {experiment_path}, got {ckpt_files}"
    best_ckpt = ckpt_files[0]
    print(f"Loading best-val_rmse checkpoint: {best_ckpt}")


    #load model from checkpoint
    model = ConvLSTMModule.load_from_checkpoint(best_ckpt, **config["model"],
                                          kernel_size=(3,3),  map_location=f'cuda:{gpu}',                           
                                # sequence_length=twsa_sequence_data.input_seq_len, 
                                horizon=twsa_sequence_data.target_seq_len)
   
    
    from pytorch_lightning.utilities.model_summary import ModelSummary
    print(ModelSummary(model))

 # --------------------------------------------------------------------------------------------------------------
 # Plotting predictions of last element in validation time series
 # --------------------------------------------------------------------------------------------------------------
 #  
    
    # grace_graceclim: C_grace replaces the training C in the model's in/outputs and in every score
    grace_climatology = load_grace_climatology() if dataset == "grace_graceclim" else None
    if dataset == "grace":
        dataloader_pred = twsa_sequence_data.GRACE_dataloader()
    elif dataset == "grace_graceclim":
        dataloader_pred = twsa_sequence_data.GRACE_dataloader(twsa_climatology=grace_climatology)
    else:
        dataloader_pred = twsa_sequence_data.predict_dataloader(period=twsa_sequence_data.train_val_test_periods[dataset])

    try:  
        predictions = torch.load(os.path.join(config["experiment"]["save_dir"],"predictions.pt"))
    except FileNotFoundError:
        predictions = trainer.predict(model, dataloader_pred) # check which time period is being used, should be validation time period
        torch.save(predictions, os.path.join(config["experiment"]["save_dir"],"predictions.pt"))
    twsa_climatology = twsa_sequence_data.twsa_climatology if grace_climatology is None else grace_climatology
    climatology = twsa_climatology.values.astype(np.float32) # raw training C (or C_grace) [12, H, W] in cm

    # get mask to mask antarctica out in the evaluation
    antarctica_mask =  twsa_sequence_data.get_landmask_without_antarctica()
    #climatology = climatology[:,antarctica_mask]  # [12, N]

    # cos(lat) area weights of the cells calculate_per_timestep_scores keeps, in its row-major order
    lat = twsa_sequence_data.data_train.ds_timeslice['lat'].values
    lat_weights = torch.cos(torch.deg2rad(torch.Tensor(np.broadcast_to(lat[:, None], antarctica_mask.shape)[antarctica_mask])))  # [N]
    metrics_dict = {'rmse': _rmse,
                    'rmse_latw': lambda y_hat, y_true: _rmse(y_hat, y_true, lat_weights=lat_weights),
                    'exp_var': torchmetrics.regression.ExplainedVariance(multioutput='raw_values')}

    per_timestep_scores = {'rmse': [], 'rmse_deseas': [], 'rmse_clim': [],
                               'rmse_latw': [], 'rmse_latw_deseas': [], 'rmse_latw_clim': [],
                               'exp_var': [], 'exp_var_deseas': [], 'exp_var_clim': []
                               }
    basin_per_timestep_scores = {'rmse': [], 'rmse_deseas': [], 'rmse_clim': [],
                               'exp_var': [], 'exp_var_deseas': [], 'exp_var_clim': []
                               }
    pixel_kge_scores = []
    pixel_nse_scores = []
    basin_kge_scores = []
    clim_basin_kge_scores = []
    skill_windows = []  # (pred, true, climatology) per window, each [T, N]

    horizon = twsa_sequence_data.target_seq_len
    
    # iterate over the batches in the predictions for the evaluations
    i = 0
    for batch in predictions:
        twsa_pred_batch = batch[0] # shape: [batchsize, output_seq_len, H, W]
        twsa_true_batch = batch[1] # shape: [batchsize, output_seq_len, H, W]
        input_sequence = batch [2] # shape: [batchsize, input_seq_len, features, H, W]
        

        # get months for each input in the sequence
        months_input_sequences = recover_month_index(batch[2][:, :,-1], batch[2][:, :,-2]) # last element (-1) is sine encoding, -2 is cos encoding
        months_output_sequences = torch.stack([(months_input_sequences[:,-1] +i +1)%12 for i in range(output_seq_len)], dim=1)



        ssims_zeros = []
        ssims_clim = []
        ssims_pred_NoClim = []
        ssims_pred_normal = []
      


        # iterate over the batch to evaluate each forecast item
        # twsa_pred_batch: [B, T, H, W], twsa_true_batch: [B, T, H, W], months_output_sequences: [B, T, H, W]
        for twsa_pred, twsa_true, months in zip(twsa_pred_batch, twsa_true_batch, months_output_sequences):

            # predictions with seasonality removed
            twsa_pred_deseas = twsa_pred-climatology[months[:,0,0]] # get the right months to subtract climatology (months are the same for all nodes so get month for node 0)
            twsa_true_deseas = twsa_true-climatology[months[:,0,0]]

            # --------------------------------------------------------------------------------------------------------------
            # RMSE scores -- per timestep/averaged over all land nodes
            # --------------------------------------------------------------------------------------------------------------  

            per_timestep_scores = calculate_per_timestep_scores(twsa_pred, 
                                                                twsa_true, 
                                                                climatology[months[:,0,0]], 
                                                                per_timestep_scores,
                                                                metrics_dict, 
                                                                lsm=antarctica_mask)
            skill_windows.append((twsa_pred[:, antarctica_mask], twsa_true[:, antarctica_mask],
                                  torch.from_numpy(climatology[months[:, 0, 0]])[:, antarctica_mask]))

        


            # --------------------------------------------------------------------------------------------------------------
            # Explained Variance scores -- pixelwise
            # --------------------------------------------------------------------------------------------------------------  
            if False:
                plot_explained_variance(twsa_pred, 
                                        twsa_true, 
                                        twsa_sequence_data, 
                                        save_dir=os.path.join(config["experiment"]["save_dir"], f'ExpVar/expVar{i}.png'))
                
                
                # # without seasonality
                plot_explained_variance(twsa_pred_deseas, 
                                        twsa_true_deseas, 
                                    twsa_sequence_data, 
                                    save_dir=os.path.join(config["experiment"]["save_dir"], f'ExpVar/NoClimExpVar{i}.png'))
            
         # --------------------------------------------------------------------------------------------------------------
            # Plot predictions
            # --------------------------------------------------------------------------------------------------------------  
            if False:
                for time_step in range(twsa_pred.shape[0]):
                    
                    plot_prediction(twsa_pred[time_step], twsa_true[time_step], twsa_sequence_data, antarctica=True, 
                                    save_dir=os.path.join(config["experiment"]["save_dir"], f"Predictions/Seq_{i}_timestep_{time_step}"))


            # --------------------------------------------------------------------------------------------------------------
            # NSE scores -- per pixel
            # --------------------------------------------------------------------------------------------------------------  

            #calculate NSE (versus long term mean)
            if False:
                pixel_nse = nse(twsa_pred, twsa_true).cpu()
                pixel_nse_scores.append(pixel_nse)
                # clim_pixel_nse = nse(torch.Tensor(climatology[months[:,0]]), twsa_true).cpu()
                # clim_pixel_nse_scores.append(clim_pixel_nse)
                if True:
                    plot_NSE_or_KGE(pixel_nse, twsa_sequence_data, 
                                    save_dir=os.path.join(config["experiment"]["save_dir"], f'NSE/Vmin0NSE{i}.png'), vmin=0)
                    # plot_NSE_or_KGE(clim_pixel_nse, twsa_sequence_data, 
                    #                 save_dir=os.path.join(config["experiment"]["save_dir"], f'NSE/NoVminClimNSE{i}.png'), vmin=-100)
            
            
            
            


            # --------------------------------------------------------------------------------------------------------------
            # SSIM scores -- per timestep because captures the structural similarity of the prediction to the ground truth
            # --------------------------------------------------------------------------------------------------------------  
            # calculate and plot SSIM for sequence with removed climatology/seasonality
            if False:
                f, ax, ssim_clim, ssim_zeros_clim, _ = plot_ssim(twsa_pred_deseas, twsa_true_deseas, 
                                                                        twsa_sequence_data,
                                                                    antarctica_mask=antarctica_mask, 
                                                                    index=i,
                            save_dir=os.path.join(config["experiment"]["save_dir"], f'SSIM/NoClim_ssim_{i}.png'))
                
                # # calculate and plot SSIM for forecast sequence and compare to pure climatology forecast SSIM
                f_clim, ax_clim, ssim, ssim_zeros, ssim_climvstrue = plot_ssim(twsa_pred, twsa_true, twsa_sequence_data,
                                                                        antarctica_mask=antarctica_mask,
                                                                            index=i,
                                                                    climatology=climatology[months[:,0,0]],
                                                save_dir=os.path.join(config["experiment"]["save_dir"], f'SSIM/ssim_vs_clim_{i}.png'))
            # ssims_zeros.append(ssim_zeros)
            # ssims_clim.append(ssim_climvstrue.cpu().numpy())
            # ssims_pred_normal.append(ssim.cpu().numpy())
            i += 1

            

    # --------------------------------------------------------------------------------------------------------------
    # save mean scores for forecasting time steps-- rmse, mape, explained variance
    # --------------------------------------------------------------------------------------------------------------  

    MEAN_per_timestep_scores = {}
    STD_per_timestep_scores = {}

    
    
    
    for key, metric_list in per_timestep_scores.items():
        # convert to numpy and stack and calculate mean over all the batches
        MEAN_per_timestep_scores[key] = np.stack(metric_list).mean(axis=0).tolist()
        STD_per_timestep_scores[key] = np.stack(metric_list).std(axis=0).tolist()

    with open(os.path.join(config["experiment"]["save_dir"], "mean_timestep_scores.json"), "w") as f:
            json.dump(MEAN_per_timestep_scores, f, indent=4)
    with open(os.path.join(config["experiment"]["save_dir"], "std_timestep_scores.json"), "w") as f:
            json.dump(STD_per_timestep_scores, f, indent=4)

    # save scores to csv, laid out like evaluation_HiGNN.py's
    pred_all, true_all, clim_all = (torch.stack(a).permute(0, 2, 1) for a in zip(*skill_windows))  # [windows, N, T]
    per_timestep_df = pd.DataFrame({'rmse': MEAN_per_timestep_scores['rmse'],
                                    'rmse_latw': MEAN_per_timestep_scores['rmse_latw'],
                                    'clim_rmse': MEAN_per_timestep_scores['rmse_clim'],
                                    'clim_rmse_latw': MEAN_per_timestep_scores['rmse_latw_clim']},
                                   index=np.arange(horizon))
    per_timestep_df = per_timestep_df.assign(**per_lead_skill_scores(pred_all, true_all, clim_all, lat_weights))
    per_timestep_df.index.name = 'Timestep'
    per_timestep_df.to_csv(os.path.join(config["experiment"]["save_dir"], "per_timestep_scores.csv"))


    
    # --------------------------------------------------------------------------------------------------------------
    # save deseasonalized KGE scores
    # --------------------------------------------------------------------------------------------------------------
# 
    # MEAN_pixel_kge = np.stack(pixel_kge_scores).mean(axis=0) # average over all batches

    # MEAN_basin_kge_scores = np.stack(basin_kge_scores).mean(axis=0)

    # MEAN_clim_basin_kge_scores = np.stack(clim_basin_kge_scores).mean(axis=0)

    # kge_scores_dict = {
    #     "mean_pixel_kge": np.nanmean(MEAN_pixel_kge, axis=1)[0].tolist(),
    #     "mean_pixel_corr": np.nanmean(MEAN_pixel_kge, axis=1)[1].tolist(), ## Does not make much sense to report this as these means won't add up to the KGE
    #     "mean_pixel_alpha": np.nanmean(MEAN_pixel_kge, axis=1)[2].tolist(),
    #     "mean_pixel_beta": np.nanmean(MEAN_pixel_kge, axis=1)[3].tolist(),

    #    # "mean_kge_per_basin": MEAN_basin_kge_scores[0].tolist(),
    #     # "mean_corr_per_basin": basin_MEAN_kge_scores[1].tolist(),
    #     # "mean_alpha_per_basin": basin_MEAN_kge_scores[2].tolist(),
    #     # "mean_beta_per_basin": basin_MEAN_kge_scores[3].tolist(),

    #     # "mean_clim_basin_kge": MEAN_clim_basin_kge_scores[0].tolist(),
    # }

    # with open(os.path.join(config["experiment"]["save_dir"], "deseasonnalized_KGE_scores.json"), "w") as f:
    #         json.dump(kge_scores_dict, f, indent=4)

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
        "--dataset", help="Dataset to evaluate on (test, val, grace or grace_graceclim)", default="test", type=str)

    args = parser.parse_args()

    evaluate(args.exp_path, gpu=args.gpu, dataset=args.dataset)


if __name__ == "__main__":
    start()
