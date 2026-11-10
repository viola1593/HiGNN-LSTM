import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric as pyg
import wandb
import yaml


# Function to load yaml configuration file
def read_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config

def save_config(config, path):
    with open(path, 'w') as file:
        yaml.dump(config, file)

def plot_graph(edge_index, pos_lat_lon, title=None, latlon_dest=None, vmax=None, edge_attr=None):
    """
    Plot flattened global graph
    Taken from neural lam.
    """
    fig, axis = plt.subplots(figsize=(10, 8), dpi=200)  # W,H
    
    if pyg.utils.is_undirected(edge_index):
        # Keep only 1 direction of edge_index
        
        keep = edge_index[0] < edge_index[1]  # (M,)
        edge_index = edge_index[:, keep]  # (2, M/2)
        if edge_attr is not None:
            # Also subset edge attributes
            edge_attr = edge_attr[keep]
    
    
    # Make lon x-axis
    pos = np.stack((pos_lat_lon[:, 0], pos_lat_lon[:, 1]), axis=1)
    if latlon_dest is None:
        pos_dest = pos     
    else:
        pos_dest = np.stack((latlon_dest[:, 0], latlon_dest[:, 1]), axis=1)
    
   # Move all to cpu and numpy, compute (in)-degrees
    degrees = (pyg.utils.degree(edge_index[1], num_nodes=pos_dest.shape[0]).cpu().numpy())
    edge_index = edge_index.cpu().numpy()


    # Plot edges
    from_pos = pos[edge_index[0]]  # (M/2, 2)
    to_pos = pos_dest[edge_index[1]]  # (M/2, 2)
    edge_lines = np.stack((from_pos, to_pos), axis=1)
    if edge_attr is not None:
        # Normalize edge weights for visualization
        edge_attr = edge_attr.cpu().numpy()
        widths = 0.1 + 2.0 * (edge_attr - edge_attr.min()) / (edge_attr.max() - edge_attr.min())
    else:
        widths = 0.4
    axis.add_collection(
        matplotlib.collections.LineCollection(
            edge_lines, lw=widths, colors="black", zorder=1
        )
    )
    
    # Plot source nodes in red
    if latlon_dest is not None:
        axis.scatter(pos[:, 0], pos[:, 1],
                    c='red', s=1, marker=".", zorder=2,
                    label='Grid nodes', alpha=0.5)
        
    # Plot destination nodes
    node_scatter = axis.scatter(pos_dest[:, 0], pos_dest[:, 1],
                                c=degrees,s=10,marker="o",
                                zorder=3,cmap="viridis", clim=[0, vmax] if vmax else None,
                                label='Basins')
        
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    plt.colorbar(node_scatter, aspect=50, extend='max' if vmax else 'neither')
    if title is not None:
        axis.set_title(title)
    return fig, axis

def great_circle_points(lat1, lon1, lat2, lon2, num_points=50, radius=1.0):
    """Generate points along the great-circle path between two lat/lon points."""
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Cartesian coordinates
    x1, y1, z1 = (math.cos(lat1)*math.cos(lon1),
                  math.cos(lat1)*math.sin(lon1),
                  math.sin(lat1))
    x2, y2, z2 = (math.cos(lat2)*math.cos(lon2),
                  math.cos(lat2)*math.sin(lon2),
                  math.sin(lat2))

    # Angle between points
    dot = x1*x2 + y1*y2 + z1*z2
    dot = max(min(dot, 1.0), -1.0)  # clamp for numerical safety
    omega = math.acos(dot)
    if abs(omega) < 1e-10:  # same point → no arc needed
        return np.tile([radius*x1, radius*y1, radius*z1], (num_points, 1))

    # Interpolate along the great circle
    points = []
    for t in np.linspace(0, 1, num_points):
        sin_omega = math.sin(omega)
        A = math.sin((1-t)*omega) / sin_omega
        B = math.sin(t*omega) / sin_omega
        x = A*x1 + B*x2
        y = A*y1 + B*y2
        z = A*z1 + B*z2
        points.append((radius*x, radius*y, radius*z))
    return np.array(points)


# Function to convert lat/lon to 3D coords on sphere
def latlon_to_xyz(lat, lon, radius=1):
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z


def plot_graph_3d(pos_lon_lat, edge_index, save_dir=None):
    
    lons, lats = pos_lon_lat[:, 0], pos_lon_lat[:, 1]


    # Create 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(-np.pi/2, np.pi/2, 100)
    x_sphere = np.cos(v)[:, None] * np.cos(u)
    y_sphere = np.cos(v)[:, None] * np.sin(u)
    z_sphere = np.sin(v)[:, None] * np.ones_like(u)
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightblue', alpha=0.3)

    # Plot nodes
    pos_xyz = {}
    for n, (lon, lat) in enumerate(zip(lons, lats)):
        x, y, z = latlon_to_xyz(lat, lon)
        pos_xyz[n] = (x, y, z)
        ax.scatter(x, y, z, color='b', s=15)


        # Plot great-circle edges
    n_nodes = len(lons)
    for i, j in edge_index.transpose(1,0).numpy():
        if i<n_nodes and j<n_nodes:
            arc_points = great_circle_points(
                lats[i].item(), lons[i].item(),
                lats[j].item(), lons[j].item(),
                num_points=50
            )
            ax.plot(arc_points[:,0], arc_points[:,1], arc_points[:,2], color='black')
    
    ax.set_axis_off()
    ax.view_init(elev=0, azim=-90)

    if save_dir is not None:
        plt.savefig(save_dir)
        plt.close()

def plot_prediction(y_hat, y_true, y_last, coords, save_dir=None, trainer=None, name='0', vmax=100):
    """
    Plotting y_last, y_hat, y_true and absolute difference between y_hat and y_true on the coords of the grid cells.
    Args:
        y_hat: predicted TWSA
        y_true: true TWSA
        y_last: last TWSA
        coords: coordinates of the grid cells to scatter plot the TWSA
        save_dir: path to save the plot
        trainer: pl.Trainer object to log the plot with wandb
        name: name of the plot
    """

    lon, lat = coords[:,0], coords[:,1]
    
    f, (ax0, ax1, ax2, ax3) = plt.subplots(4,1,figsize=(5,10))
    p0 = ax0.scatter(lon, lat, c=y_last, cmap='coolwarm', s=1, vmin=-vmax, vmax=vmax)
    p1 = ax1.scatter(lon, lat, c=y_hat, cmap='coolwarm', s=1,vmin=-vmax, vmax=vmax)
    p2 = ax2.scatter(lon, lat, c=y_true, cmap='coolwarm', s=1,vmin=-vmax, vmax=vmax)
    ape = (abs(y_hat-y_true)) 
    p3 = ax3.scatter(lon, lat, c=ape, cmap='Reds',s=1, vmax=10)
    #p3 = ax3.imshow(abs(y_hat-y_true)/(y_true+ 1e-5), cmap='Reds', vmin=0, vmax=1)
    ax0.set_title('Last TWSA')
    ax1.set_title('Predicted TWSA')
    ax2.set_title('True TWSA')
    ax3.set_title('Absolute Difference')
    f.suptitle('TWSA Prediction on Train Set')
    
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
    return f


import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.dates import date2num
import pandas as pd
from datetime import datetime

def plot_rmse_broken_axis(
    dates,           # array of datetime objects or strings, length N
    rmse,            # array of shape (6, N) – one row per lead time
    gap_start_id=None, # index of last date BEFORE gap; None -> series is contiguous
    gap_end_id=None,   # index of first date AFTER gap
    lead_labels=None,
    ylabel="RMSE (cm EWH)",
    title=None,
    cmap_name="Blues_r",
    figsize=(12, 4),
    savepath=None,
):
    """
    Plot RMSE per forecast month with a broken x-axis to represent a data gap.
 
    Parameters
    ----------
    dates        : array-like of datetime | length N
    rmse         : np.ndarray, shape (n_leads, N)
    gap_start_id : int | None – last index of the pre-gap segment (inclusive).
                   None plots the whole series on a single axis, without a break.
    gap_end_id   : int  – first index of the post-gap segment (inclusive)
    lead_labels  : list of str, optional
    """
    dates = pd.to_datetime(dates)
    rmse  = np.asarray(rmse)
    n_leads, N = rmse.shape
    assert len(dates) == N, "dates and rmse second dimension must match"
 
    if lead_labels is None:
        lead_labels = [f"Lead {i+1}" for i in range(n_leads)]
 
    # ── split into pre- and post-gap segments ──────────────────────────────
    if gap_start_id is None:
        segments = [(dates, rmse)]
    else:
        pre_idx  = slice(0, gap_start_id + 1)      # 0 … gap_start_id  (inclusive)
        post_idx = slice(gap_end_id, N)             # gap_end_id … end  (inclusive)
        segments = [(dates[pre_idx],  rmse[:, pre_idx]),
                    (dates[post_idx], rmse[:, post_idx])]

    # width ratio proportional to number of months shown
    width_ratios = [len(seg_dates) for seg_dates, _ in segments]

    # ── colour map: one colour per lead ────────────────────────────────────
    cmap   = plt.get_cmap(cmap_name)
    colors = [cmap(i / (n_leads - 1 if n_leads > 1 else 1)) for i in range(n_leads)]
 
    # ── figure ─────────────────────────────────────────────────────────────
    plt.rcParams.update({
        # "font.family":      "serif",
        # "font.serif":       ["Times New Roman", "DejaVu Serif"],
        "font.size":        11,
        "axes.linewidth":   0.8,
        "xtick.direction":  "in",
        "ytick.direction":  "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    })
 
    fig, axs = plt.subplots(
        1, len(segments),
        figsize=figsize,
        sharey=True,
        squeeze=False,
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.04},
    )
    axs = list(axs[0])
    ax1 = axs[0]

    # ── plot the segments ──────────────────────────────────────────────────
    for ax, (seg_dates, seg_rmse) in zip(axs, segments):
        for lead_i in range(n_leads):
            ax.plot(seg_dates, seg_rmse[lead_i], color=colors[lead_i], linewidth=1.3,
                    alpha=0.9, label=lead_labels[lead_i])

    # ── x-axis formatting ──────────────────────────────────────────────────
    for ax, (dts, _) in zip(axs, segments):
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        ax.set_xlim(dts[0] - pd.Timedelta(days=20),
                    dts[-1] + pd.Timedelta(days=20))
 
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
 
    # ── broken-axis diagonal marks ─────────────────────────────────────────
    if len(segments) == 2:
        ax2   = axs[1]
        d      = 0.015   # size of diagonal tick marks in axes coordinates
        kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False, linewidth=0.9)
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)   # bottom-right of ax1
        ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # top-right of ax1

        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (-d, +d), **kwargs)          # bottom-left of ax2
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)   # top-left of ax2

        # hide inner spines to sell the break
        ax1.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.tick_params(left=False)   # no y-ticks on right panel

    # ── labels, legend, title ──────────────────────────────────────────────
    ax1.set_ylabel(ylabel, labelpad=6)
 
    # shared x-label centred across both panels
    fig.text(0.5, -0.04, "Date", ha="center", va="top", fontsize=11)
 
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=n_leads,
        fontsize=9,
        frameon=True,
        # title="Forecast Month",
        # title_fontsize=9,
    )
 
    if title:
        fig.suptitle(title, y=1.10, fontsize=12)
 
    # gap annotation, spanning the months the break actually skips
    if len(segments) == 2:
        fig.text(
            0.01, 0.05,
            f"   ⋯\ndata gap\n{segments[0][0][-1]:%Y-%m}–{segments[1][0][0]:%Y-%m}   ",
            ha="left", va="top",
            fontsize=8.5, color="0.5", style="italic",
            transform=fig.transFigure,
        )

    fig.tight_layout()
 
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        print(f"Saved to {savepath}")
 
    return fig, tuple(axs)