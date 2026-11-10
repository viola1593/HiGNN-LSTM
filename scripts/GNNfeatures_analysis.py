"""
Simple SHAP Analysis for GNN-LSTM Model using DeepLiftSHAP

Memory-efficient script for:
1. Computing SHAP values for GNN outputs
2. Computing SHAP values for LSTM outputs
3. Plotting results

Kept for the record only: this attribution analysis was removed from the revised results, where the ablations
(results_tables.py, basin_ablation.py) and the latent removal (run_latent_removal.py) replace it.
"""

# SET GPU FIRST - BEFORE ALL OTHER IMPORTS
import os

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import glob
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple


from captum.attr import Saliency, NoiseTunnel
from models.pl_hmmodel import HMModule
from datamodules.TWSA_heterograph_datamodule import TWSA_HeteroGraphDatamodule

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {DEVICE}")


def load_model_and_data(config: Dict, checkpoint_path: str, device):
    """Load model and get a batch of data"""
        
    datamodule = TWSA_HeteroGraphDatamodule(**config['datamodule'])
    datamodule.setup(stage='fit')
    
    model = HMModule.load_from_checkpoint(
        checkpoint_path,
        **config['model'],
        num_input_features=datamodule.num_features,
        num_ocean_features=datamodule.num_ocean_features,
        use_ocean_nodes=datamodule.use_ocean_nodes,
        sequence_length=datamodule.input_seq_len,
        output_seq_len=datamodule.target_seq_len, 
        land_sea_mask=None,
        map_location=device
    )
    model = model.to(device)
    model.eval()
    
    val_dataloader = datamodule.val_dataloader()
    data_batch = next(iter(val_dataloader))
    data_batch = data_batch.to(device)
    
    return model, datamodule, config, data_batch


class RNNWrapper(torch.nn.Module):
    """
    Wraps just the RNN forward pass, taking gnn_output as input.
    Holds twsa_history and timestep_embed fixed as context.
    """
    def __init__(self, model, horizon_step):
        super().__init__()
        self.model = model
        self.horizon_step = horizon_step
    
    def forward(self, input_ts):
       
        actual_batch = input_ts.shape[0]  # n_steps*batch during IG

        
        h0 = torch.zeros(self.model.rnn_layers, actual_batch, self.model.rnn_hidden_size).to(input_ts.device)
        c0 = torch.zeros(self.model.rnn_layers, actual_batch, self.model.rnn_hidden_size).to(input_ts.device)
        
        res, _ = self.model.rnn(input_ts, (h0, c0))
        res_norm = self.model.layer_norm(self.model.act(res[:, -1]))
        res_norm = self.model.output_proj(res_norm)
        twsa_last = input_ts[:,-1,self.model.timestep_embed_dim]  # [n_steps*batch, 12, 1]
        twsa_out = res_norm.squeeze() + twsa_last.squeeze()
        return twsa_out
    
def compute_gnn_latent_importance(model, data, n_steps=25):
    """
    Returns attribution of each gnn_output dimension
    to the RNN forecast, per horizon step.
    
    Returns:
        attributions: [horizon, batch, seq_len, gnn_emb_dim]
    """
    # model.eval()
    # model.rnn.train()
    all_attributions = []
    
    # --- Run the GNN portion once, no grad needed ---
    with torch.no_grad():
        data = data.clone()
        batch_size = data['grid'].x.shape[0]
        twsa_in = data['grid'].x[:, :, model.twsa_index].unsqueeze(-1)
        
        data['grid'].x = data['grid'].x.view(batch_size, -1)
        data['ocean'].x = data['ocean'].x.view(data['ocean'].x.shape[0], -1)
        data = model.embedder(data)
        data = model.grid2mesh(data)
        twsa_history_fixed = twsa_in.clone()
    
    # --- Now attribute through RNN for each horizon step ---
    for i in range(model.horizon):
        with torch.no_grad():
            gnn_emb = data.clone()
            timestep_embed = model.timestep_embeddings(
                torch.tensor(i, device=gnn_emb['grid'].x.device)
            )
            gnn_emb, gamma_edge, beta_edge, gamma_node, beta_node = model.shared_forward(
                gnn_emb, timestep_embed
            )
            gnn_emb = model.mesh2grid(gnn_emb, gamma_edge, beta_edge, gamma_node, beta_node)
            gnn_output = model.outputter(gnn_emb)
            gnn_output = gnn_output.reshape(batch_size, model.seq_len, -1)
        
        # gnn_output needs grad for IG
        timestep_embed_expanded = timestep_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, model.seq_len, -1)
        input_ts = torch.cat((timestep_embed_expanded, twsa_history_fixed, gnn_output), dim=-1) # --> [batch_size*num_nodes, seq_len, 1+num_features+trigmonths(2)]
            
        input_ts = input_ts.detach().requires_grad_(True)
        
        # Baseline: zeros (or use mean embedding across dataset for a better baseline)
        # baseline = torch.zeros_like(input_ts[:1])
        
        # Build the RNN wrapper for this horizon step
        wrapper = RNNWrapper(model, i)
        # Wrap with NoiseTunnel
        saliency = Saliency(wrapper)
        nt       = NoiseTunnel(saliency)
        # ig = IntegratedGradients(wrapper)
        # Tune stdevs per input based on their scale
        # rule of thumb: stdevs ~ 0.1 * input.std()
        stdevs_gnn  = 0.1 * input_ts.std().item()
        # stdevs_twsa = 0.1 * twsa_input_d.std().item()
        # NoiseTunnel stacks nt_samples noisy copies into one LSTM batch; chunk so the LSTM never
        # sees more rows than in training (n_steps*batch_size rows crash cuDNN with an illegal memory access)
        chunk_size = -(-batch_size // n_steps)
        attr_sum = 0
        for chunk in input_ts.split(chunk_size):
            # Attribution shape: [chunk, seq_len, gnn_emb_dim]
            attribution = nt.attribute(
                chunk,
                nt_type='smoothgrad',       # average of gradients under noise
                nt_samples=n_steps,      # number of noisy samples
                stdevs=stdevs_gnn,
                abs=False                   # keep sign, take abs later
            )
            attr_sum = attr_sum + attribution.abs().sum(dim=[0,1])
            del attribution

        all_attributions.append((attr_sum / (batch_size * model.seq_len)).cpu())
        print(f"Horizon step {i+1}/{model.horizon} done")


        del attr_sum #, pred_input, pred_baseline, expected_diff
        torch.cuda.empty_cache()

    
    return torch.stack(all_attributions, dim=0)

def compute_attributions_over_dataset(model, val_loader, n_steps=25, device='cuda'):
    
    model.eval()
    model.rnn.train()
    
    all_attr_gnn       = []
    all_attr_twsa      = []
    all_attr_timestep  = []

    for batch_idx, data in enumerate(val_loader):
        data = data.to(device)
        
        try:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                attributions = compute_gnn_latent_importance(
                    model, data, n_steps=n_steps
                )
            
            # [horizon, 4], [horizon, 1], [horizon, 2]
            all_attr_gnn.append(attributions[:,3:])
            all_attr_twsa.append(attributions[:,2])
            all_attr_timestep.append(attributions[:,0:2])
            
        except Exception as e:
            print(f"Batch {batch_idx} failed: {e}")
            continue
        
        if batch_idx % 10 == 0:
            print(f"Processed {batch_idx}/{len(val_loader)} batches")

    # Stack and average over all batches
    # [num_batches, horizon, dim] -> mean over num_batches -> [horizon, dim]
    attr_gnn_mean      = torch.stack(all_attr_gnn).mean(dim=0)
    attr_twsa_mean     = torch.stack(all_attr_twsa).mean(dim=0)
    attr_timestep_mean = torch.stack(all_attr_timestep).mean(dim=0)

    attr_gnn_std      = torch.stack(all_attr_gnn).std(dim=0)
    attr_twsa_std     = torch.stack(all_attr_twsa).std(dim=0)
    attr_timestep_std = torch.stack(all_attr_timestep).std(dim=0)
    attr_dict = {'gnn_attr_mean': attr_gnn_mean, 
                 'gnn_attr_std': attr_gnn_std, 
                 'twsa_attr_mean': attr_twsa_mean,
                 'twsa_attr_std': attr_twsa_std,
                 'timestep_attr_mean': attr_timestep_mean,
                 'timestep_attr_std': attr_timestep_std}

    return attr_dict

def summarize_attributions(attributions):
    """
    attributions: [horizon, batch, seq_len, gnn_emb_dim]
    """
    # Mean absolute attribution per gnn_emb_dim across batch and seq_len
    # → which latent dims matter most overall
    dim_importance = attributions.abs().mean(dim=[0, 1, 2])  # [gnn_emb_dim]
    
    # How importance evolves across horizon steps
    # → does the GNN matter more for near vs far forecasts?
    horizon_importance = attributions.abs().mean(dim=[1, 2, 3])  # [horizon]
    
    # Which seq_len positions in gnn_output matter most
    # → which historical timestep's GNN embedding drives prediction
    seq_importance = attributions.abs().mean(dim=[0, 1, 3])  # [seq_len]
    
    return {
        'dim_importance': dim_importance,       # [gnn_emb_dim]
        'horizon_importance': horizon_importance, # [horizon]
        'seq_importance': seq_importance,         # [seq_len]
    }




# Function to load yaml configuration file
def read_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config

def save_config(config, path):
    with open(path, 'w') as file:
        yaml.dump(config, file)

def create_experiment_dir(exp_path: str) -> str:
    """Create feature-importance output directory inside the experiment directory.

    Args:
        exp_path: experiment directory

    Returns:
        path of the created output directory
    """
    exp_dir_path = os.path.join(exp_path, "fi_analysis")
    os.makedirs(exp_dir_path, exist_ok=True)
    return exp_dir_path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

def plot_nt_attributions(attr_gnn_mean, attr_gnn_std,
                         attr_twsa_mean, attr_twsa_std,
                         attr_timestep_mean, attr_timestep_std,
                         save_path="ig_attributions.pdf"):
    """
    All inputs: [horizon, dim] — mean and std aggregated over batch/dataset
    """
    gnn_h      = attr_gnn_mean          # [horizon, 4]
    gnn_h_std  = attr_gnn_std           # [horizon, 4]
    twsa_h     = attr_twsa_mean         # [horizon, 1]
    twsa_h_std = attr_twsa_std          # [horizon, 1]
    ts_h       = attr_timestep_mean     # [horizon, 2]
    ts_h_std   = attr_timestep_std      # [horizon, 2]

    gnn_per_horizon      = gnn_h.mean(dim=-1).numpy()
    gnn_per_horizon_std  = gnn_h_std.mean(dim=-1).numpy()
    twsa_per_horizon     = twsa_h.squeeze(-1).numpy()
    twsa_per_horizon_std = twsa_h_std.squeeze(-1).numpy()
    ts_per_horizon       = ts_h.mean(dim=-1).numpy()
    ts_per_horizon_std   = ts_h_std.mean(dim=-1).numpy()
    gnn_dim_mean         = gnn_h.mean(dim=0).numpy()   # [4]
    gnn_dim_std          = gnn_h_std.mean(dim=0).numpy()

    horizon_steps = np.arange(1, attr_gnn_mean.shape[0] + 1)
    colors_gnn    = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    color_twsa    = "#607D8B"
    color_ts      = "#9C27B0"

    fig = plt.figure(figsize=(8, 6))
    fontsize=10
    gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ── Panel 1: Grouped bar with error bars ──────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    group_means  = [attr_gnn_mean.mean().item(),
                    attr_twsa_mean.mean().item(),
                    attr_timestep_mean.mean().item()]
    group_stds   = [attr_gnn_std.mean().item(),
                    attr_twsa_std.mean().item(),
                    attr_timestep_std.mean().item()]
    group_labels = ["GNN Output", "TWSA History", "Timestep Embed"]
    group_colors = ["#2196F3", color_twsa, color_ts]

    bars = ax1.bar(group_labels, group_means, color=group_colors,
                   width=0.5, edgecolor="white", linewidth=2,
                   yerr=group_stds, capsize=8, error_kw=dict(elinewidth=2.5))
    ax1.set_ylabel("Mean Attribution", fontsize=fontsize+2,)
    ax1.set_title("(a) Overall Input Group Importance", fontsize=fontsize)
    ax1.set_ylim(0, max(m + s for m, s in zip(group_means, group_stds)) * 1.25)
    ax1.tick_params(axis='both', labelsize=fontsize)
    for label in ax1.get_xticklabels():
        label.set_fontsize(fontsize)

    # ── Panel 2: Line plot with shaded std ────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for vals, stds, label, color, marker in [
        (gnn_per_horizon,  gnn_per_horizon_std,  "GNN Output",     "#2196F3", "o"),
        (twsa_per_horizon, twsa_per_horizon_std, "TWSA History",   color_twsa, "s"),
        (ts_per_horizon,   ts_per_horizon_std,   "Timestep Embed", color_ts,   "^"),
    ]:
        ax2.plot(horizon_steps, vals, marker=marker, label=label,
                 color=color, linewidth=3, markersize=fontsize)
        ax2.fill_between(horizon_steps,
                         vals - stds,
                         vals + stds,
                         alpha=0.15, color=color)  # shaded band = ±1 std
    ax2.set_xlabel("Forecast Horizon Step", fontsize=fontsize)
    ax2.set_ylabel("Mean Attribution", fontsize=fontsize)
    ax2.set_title("(b) Attribution by Forecast Horizon", fontsize=fontsize)
    ax2.set_xticks(horizon_steps)
    ax2.tick_params(axis='both', labelsize=fontsize)
    ax2.legend(fontsize=fontsize, loc='best', framealpha=0.95)
    ax2.grid(True, alpha=0.3, linewidth=1)

    # ── Panel 3: Bar per GNN dim with error bars ──────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    dim_labels = [f"GNN Dim {i+1}" for i in range(4)]
    bars3 = ax3.bar(dim_labels, gnn_dim_mean, color=colors_gnn,
                    width=0.5, edgecolor="white", linewidth=2,
                    yerr=gnn_dim_std, capsize=fontsize,
                    error_kw=dict(elinewidth=2.5))
    ax3.set_ylabel("Mean Attribution", fontsize=fontsize+2)
    ax3.set_title("(c) Importance per GNN Latent Dimension", fontsize=fontsize)
    ax3.set_ylim(0, max(gnn_dim_mean + gnn_dim_std) * 1.25)
    ax3.tick_params(axis='both', labelsize=fontsize)
    for label in ax3.get_xticklabels():
        label.set_fontsize(fontsize)

    # ── Panel 4: Heatmap with std annotated ───────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    heatmap_data = gnn_h.numpy().T   # [4, horizon]
    heatmap_std  = gnn_h_std.numpy().T

    im = ax4.imshow(heatmap_data, aspect="auto", cmap="YlOrRd",
                    extent=[0.5, attr_gnn_mean.shape[0] + 0.5, 3.5, -0.5])

    # Annotate each cell with mean ± std
    for row in range(4):
        for col in range(attr_gnn_mean.shape[0]):
            ax4.text(col + 1, row,
                     f"{heatmap_data[row, col]:.3f}",
                    #  f"{heatmap_data[row, col]:.3f}\n±{heatmap_std[row, col]:.3f}",
                     ha='center', va='center', fontsize=fontsize, color='black')

    ax4.set_yticks(range(4))
    ax4.set_yticklabels([f"Dim {i+1}" for i in range(4)], fontsize=fontsize)
    ax4.set_xticks(horizon_steps)
    ax4.set_xticklabels(horizon_steps, fontsize=fontsize)
    ax4.set_xlabel("Forecast Horizon Step", fontsize=fontsize)
    ax4.set_title("(d) GNN Dim Attribution Across Horizon", fontsize=fontsize)
    cbar = plt.colorbar(im, ax=ax4, label="Mean Attribution")
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label("Mean Attribution", fontsize=fontsize)

    # fig.suptitle("Smooth Gradients: LSTM Input Attribution",
    #              fontsize=16, fontweight="bold", y=0.995)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

def plot_2_nt_attributions(attr_gnn_mean, attr_gnn_std,
                         attr_twsa_mean, attr_twsa_std,
                         attr_timestep_mean, attr_timestep_std,
                         save_path="ig_2attributions.pdf"):
    """
    All inputs: [horizon, dim] — mean and std aggregated over batch/dataset
    """
    gnn_h      = attr_gnn_mean          # [horizon, 4]
    gnn_h_std  = attr_gnn_std           # [horizon, 4]
    twsa_h     = attr_twsa_mean         # [horizon, 1]
    twsa_h_std = attr_twsa_std          # [horizon, 1]
    ts_h       = attr_timestep_mean     # [horizon, 2]
    ts_h_std   = attr_timestep_std      # [horizon, 2]

    gnn_per_horizon      = gnn_h.mean(dim=-1).numpy()
    gnn_per_horizon_std  = gnn_h_std.mean(dim=-1).numpy()
    twsa_per_horizon     = twsa_h.squeeze(-1).numpy()
    twsa_per_horizon_std = twsa_h_std.squeeze(-1).numpy()
    ts_per_horizon       = ts_h.mean(dim=-1).numpy()
    ts_per_horizon_std   = ts_h_std.mean(dim=-1).numpy()
    gnn_dim_mean         = gnn_h.mean(dim=0).numpy()   # [4]
    gnn_dim_std          = gnn_h_std.mean(dim=0).numpy()

    horizon_steps = np.arange(1, attr_gnn_mean.shape[0] + 1)
    colors_gnn    = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    color_twsa    = "#607D8B"
    color_ts      = "#9C27B1"

    fig = plt.figure(figsize=(8, 5))
    fontsize=10
    gs  = gridspec.GridSpec(1, 2, hspace=0.35, wspace=0.3)

    
    # ── Panel 2: Line plot with shaded std ────────────────────────
    ax2 = fig.add_subplot(gs[0, 0])
    for vals, stds, label, color, marker in [
        (gnn_per_horizon,  gnn_per_horizon_std,  "GNN Output",     "#2196F3", "o"),
        (twsa_per_horizon, twsa_per_horizon_std, "TWSA History",   color_twsa, "s"),
        (ts_per_horizon,   ts_per_horizon_std,   "Embedded timestep", color_ts,   "^"),
    ]:
        ax2.plot(horizon_steps, vals, marker=marker, label=label,
                 color=color, linewidth=3, markersize=fontsize)
        ax2.fill_between(horizon_steps,
                         vals - stds,
                         vals + stds,
                         alpha=0.15, color=color)  # shaded band = ±1 std
    ax2.set_xlabel("Forecast Horizon Step", fontsize=fontsize)
    ax2.set_ylabel("Mean Attribution", fontsize=fontsize)
    ax2.set_title("(a) Attribution by Forecast Horizon", fontsize=fontsize)
    ax2.set_xticks(horizon_steps)
    ax2.tick_params(axis='both', labelsize=fontsize)
    ax2.legend(fontsize=fontsize, loc='best', )
    ax2.grid(True, alpha=0.3, linewidth=1)

    # ── Panel 3: Bar per GNN dim with error bars ──────────────────
    ax3 = fig.add_subplot(gs[0, 1])
    dim_labels = [f"GNN Dim {i+1}" for i in range(4)]
    bars3 = ax3.bar(dim_labels, gnn_dim_mean, color=colors_gnn,
                    width=0.5, edgecolor="white", linewidth=2,
                    yerr=gnn_dim_std, capsize=fontsize,
                    error_kw=dict(elinewidth=2.5))
    ax3.set_ylabel("Mean Attribution", fontsize=fontsize)
    ax3.set_title("(b) Importance per GNN Latent Dimension", fontsize=fontsize)
    ax3.set_ylim(0, max(gnn_dim_mean + gnn_dim_std) * 1.25)
    ax3.tick_params(axis='both', labelsize=fontsize)
    for label in ax3.get_xticklabels():
        label.set_fontsize(fontsize)


    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

def plot_1_nt_attributions(attr_gnn_mean, attr_gnn_std,
                         attr_twsa_mean, attr_twsa_std,
                         attr_timestep_mean, attr_timestep_std,
                         save_path="ig_1attributions.pdf"):
    """
    All inputs: [horizon, dim] — mean and std aggregated over batch/dataset
    """
    gnn_h      = attr_gnn_mean          # [horizon, 4]
    gnn_h_std  = attr_gnn_std           # [horizon, 4]
    twsa_h     = attr_twsa_mean         # [horizon, 1]
    twsa_h_std = attr_twsa_std          # [horizon, 1]
    ts_h       = attr_timestep_mean     # [horizon, 2]
    ts_h_std   = attr_timestep_std      # [horizon, 2]

    gnn_per_horizon      = gnn_h.mean(dim=-1).numpy()
    gnn_per_horizon_std  = gnn_h_std.mean(dim=-1).numpy()
    twsa_per_horizon     = twsa_h.squeeze(-1).numpy()
    twsa_per_horizon_std = twsa_h_std.squeeze(-1).numpy()
    ts_per_horizon       = ts_h.mean(dim=-1).numpy()
    ts_per_horizon_std   = ts_h_std.mean(dim=-1).numpy()
    gnn_dim_mean         = gnn_h.mean(dim=0).numpy()   # [4]
    gnn_dim_std          = gnn_h_std.mean(dim=0).numpy()

    horizon_steps = np.arange(1, attr_gnn_mean.shape[0] + 1)
    colors_gnn    = plt.cm.tab20b.colors[-4:]  # 4 distinct colors for GNN dims
    color_twsa    =  "#2275D4"
    color_ts      = 'grey' #"#42484B"

    fig = plt.figure(figsize=(6, 5))
    fontsize=11
    gs  = gridspec.GridSpec(1, 1, hspace=0.35, wspace=0.3)

    
    # ── Panel 2: Line plot with shaded std ────────────────────────
    ax2 = fig.add_subplot(gs[0, 0])
    gnn_dims = [(gnn_h[:, i], gnn_h_std[:, i], f"GNN Dim {i+1}", color, "o") for i, color in enumerate(colors_gnn)]
    for vals, stds, label, color, marker in [
        *gnn_dims,
        # (gnn_per_horizon,  gnn_per_horizon_std,  "GNN Output",     "#2196F3", "o"),
        (twsa_per_horizon, twsa_per_horizon_std, "TWSA history",   color_twsa, "s"),
        (ts_per_horizon,   ts_per_horizon_std,   "Embedded timestep", color_ts,   "^"),
    ]:
        ax2.plot(horizon_steps, vals, marker=marker, label=label,
                 color=color, linewidth=3, markersize=fontsize)
        ax2.fill_between(horizon_steps,
                         vals - stds,
                         vals + stds,
                         alpha=0.15, color=color)  # shaded band = ±1 std
    ax2.set_xlabel("Lead time (months)", fontsize=fontsize)
    ax2.set_ylabel("Mean attribution", fontsize=fontsize)
    ax2.set_title("Attribution by lead time", fontsize=fontsize)
    ax2.set_xticks(horizon_steps)
    ax2.tick_params(axis='both', labelsize=fontsize)
    ax2.legend(fontsize=fontsize-1, loc='best', )
    ax2.grid(True, alpha=0.3, linewidth=1)

 
    
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_ig_attributions(attr_gnn, attr_twsa, attr_timestep, save_path="ig_attributions.pdf"):
    """
    attr_gnn:       [horizon, batch, seq_len, 4]
    attr_twsa:      [horizon, batch, seq_len, 1]
    attr_timestep:  [horizon, batch, seq_len, 2]
    """
    # --- Aggregate over batch and seq_len ---
    # Mean absolute attribution: [horizon, dim]
    gnn_h     = attr_gnn.abs() #.mean(dim=[1, 2])       # [horizon, 4]
    twsa_h    = attr_twsa.abs() #.mean(dim=[1, 2])       # [horizon, 1]
    ts_h      = attr_timestep.abs() #.mean(dim=[1, 2])   # [horizon, 2]

    # Collapse to scalars per input group
    gnn_per_horizon  = gnn_h.mean(dim=-1).numpy()      # [horizon]
    twsa_per_horizon = twsa_h.squeeze(-1).numpy()      # [horizon]
    ts_per_horizon   = ts_h.mean(dim=-1).numpy()       # [horizon]

    # Overall mean per GNN dim across all horizon steps
    gnn_dim_mean = gnn_h.mean(dim=0).numpy()           # [4]

    horizon_steps = np.arange(1, attr_gnn.shape[0] + 1)

    # ── Figure layout ──────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    colors_gnn = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]  # one per dim
    color_twsa = "#607D8B"
    color_ts   = "#9C27B0"

    # ── Panel 1: Grouped bar — overall input group importance ──────
    ax1 = fig.add_subplot(gs[0, 0])
    group_means = [
        attr_gnn.abs().mean().item(),
        attr_twsa.abs().mean().item(),
        attr_timestep.abs().mean().item()
    ]
    group_labels = ["GNN Output", "TWSA History", "Timestep Embed"]
    group_colors = ["#2196F3", color_twsa, color_ts]
    bars = ax1.bar(group_labels, group_means, color=group_colors, width=0.5, edgecolor="white")
    ax1.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax1.set_ylabel("Mean |Attribution| (TWSA units)")
    ax1.set_title("(a) Overall Input Group Importance")
    ax1.set_ylim(0, max(group_means) * 1.25)

    # ── Panel 2: Line plot — attribution per horizon step ──────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(horizon_steps, gnn_per_horizon,  marker="o", label="GNN Output",     color="#2196F3", linewidth=2)
    ax2.plot(horizon_steps, twsa_per_horizon, marker="s", label="TWSA History",   color=color_twsa, linewidth=2)
    ax2.plot(horizon_steps, ts_per_horizon,   marker="^", label="Timestep Embed", color=color_ts, linewidth=2)
    ax2.set_xlabel("Forecast Horizon Step")
    ax2.set_ylabel("Mean |Attribution| (TWSA units)")
    ax2.set_title("(b) Attribution by Forecast Horizon")
    ax2.set_xticks(horizon_steps)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Bar — per GNN dim importance ─────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    dim_labels = [f"GNN Dim {i+1}" for i in range(4)]
    bars3 = ax3.bar(dim_labels, gnn_dim_mean, color=colors_gnn, width=0.5, edgecolor="white")
    ax3.bar_label(bars3, fmt="%.4f", padding=3, fontsize=9)
    ax3.set_ylabel("Mean |Attribution| (TWSA units)")
    ax3.set_title("(c) Importance per GNN Latent Dimension")
    ax3.set_ylim(0, gnn_dim_mean.max() * 1.25)

    # ── Panel 4: Heatmap — GNN dim importance across horizon ──────
    ax4 = fig.add_subplot(gs[1, 1])
    heatmap_data = gnn_h.numpy().T  # [4, horizon]
    im = ax4.imshow(heatmap_data, aspect="auto", cmap="YlOrRd",
                    extent=[0.5, attr_gnn.shape[0] + 0.5, 3.5, -0.5])
    ax4.set_yticks(range(4))
    ax4.set_yticklabels([f"Dim {i+1}" for i in range(4)])
    ax4.set_xticks(horizon_steps)
    ax4.set_xlabel("Forecast Horizon Step")
    ax4.set_title("(d) GNN Dim Attribution Across Horizon")
    plt.colorbar(im, ax=ax4, label="|Attribution|")

    fig.suptitle("Integrated Gradients: LSTM Input Attribution", fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
    print(f"Saved to {save_path}")



def main(args):
    pl.seed_everything(42, workers=True)

    config = read_config(os.path.join(args.exp_path,  'config.yaml'))
    config['pl']['devices'] = [args.gpu]
    # config['datamodule']['batch_size']=1

    # Recursively search for .ckpt files in all subfolders
    ckpt_files = glob.glob(os.path.join(args.exp_path, "**", "*.ckpt"), recursive=True)
    
    if ckpt_files:
        # Use the earliest checkpoint (by modification time) (last checkpoint is also saved)
        latest_ckpt = min(ckpt_files, key=os.path.getmtime)
        print(f"Found checkpoint: {latest_ckpt}")
    else:
        print("No checkpoint found.")
        return

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading model and data...")
    model, datamodule, config, data_batch = load_model_and_data(config, latest_ckpt, device=device)
    
    num_features = model.model.num_features
    gnn_embedding_dim = model.model.gnn_embedding_dim

    output_dir=create_experiment_dir(args.exp_path)
    
    print(f"✓ Model loaded")
    print(f"  - Num features: {num_features}")
    print(f"  - GNN embedding dim: {gnn_embedding_dim}")
    print(f"  - Batch size: {data_batch['grid'].x.shape[0]}")
    
    # Compute SHAP values
    torch.cuda.empty_cache()  # Clear GPU memory
    
    # comput attributitions if they are not saved already
    if os.path.exists(os.path.join(output_dir, 'attr_dict.pkl')):
        print("Loading precomputed attributions from disk...")
        with open(os.path.join(output_dir, 'attr_dict.pkl'), 'rb') as f:
            attr_dict = torch.load(f)
        attr_gnn_cpu      = attr_dict['gnn_attr_mean']
        attr_twsa_cpu     = attr_dict['twsa_attr_mean']
        attr_timestep_cpu = attr_dict['timestep_attr_mean']
        attr_gnn_cpu_std     = attr_dict['gnn_attr_std']
        attr_twsa_cpu_std     = attr_dict['twsa_attr_std']
        attr_timestep_cpu_std = attr_dict['timestep_attr_std']
    else:
        attr_dict = compute_attributions_over_dataset(model.model, datamodule.val_dataloader(),
                                              n_steps=10, device=device)

        attr_gnn_cpu      = attr_dict['gnn_attr_mean'].cpu().float()
        attr_twsa_cpu     = attr_dict['twsa_attr_mean'].cpu().float()
        attr_timestep_cpu = attr_dict['timestep_attr_mean'].cpu().float()
        attr_gnn_cpu_std     = attr_dict['gnn_attr_std'].cpu().float()
        attr_twsa_cpu_std     = attr_dict['twsa_attr_std'].cpu().float()
        attr_timestep_cpu_std = attr_dict['timestep_attr_std'].cpu().float()
        # create cpu dict to save
        attr_dict = {'gnn_attr_mean': attr_gnn_cpu, 
                     'gnn_attr_std': attr_gnn_cpu_std,
                     'twsa_attr_mean': attr_twsa_cpu,
                     'twsa_attr_std': attr_twsa_cpu_std,
                     'timestep_attr_mean': attr_timestep_cpu,
                     'timestep_attr_std': attr_timestep_cpu_std}

        # save attr_dict to disk for later plotting without recomputation
        with open(os.path.join(output_dir, 'attr_dict.pkl'), 'wb') as f:
            torch.save(attr_dict, f)

    
    # timestep_attr = attributions[:,:,:,0:2]
    # twsa_attr = attributions[:,:,:,2]
    # gnn_attr = attributions[:,:,:,3:]
    mean_target_value = data_batch['grid'].y.abs().mean()

    


    
    # plot_ig_attributions(attr_gnn_cpu, attr_twsa_cpu, attr_timestep_cpu,
    #                      save_path=os.path.join(output_dir, 'smoothgrad_attrs_valdataset.png'))
    plot_1_nt_attributions(attr_gnn_cpu, attr_gnn_cpu_std,
                         attr_twsa_cpu, attr_twsa_cpu_std,
                         attr_timestep_cpu, attr_timestep_cpu_std,
                         save_path=os.path.join(output_dir, 'smoothgrad_attrs_valdataset1.png'))


    
    
    
   


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='Simple SHAP Analysis using DeepLift')
    parser.add_argument('--exp_path', type=str, required=True, help='Experiment path')
    parser.add_argument('--gpu', type=int, required=True, help='Which GPU to use')
    
    
    args = parser.parse_args()
    main(args)
