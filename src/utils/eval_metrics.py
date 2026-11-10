"""Pure metric helpers shared by the evaluation scripts.

Extracted from scripts/evaluation_HiGNN.py, scripts/compare_models.py and
scripts/evaluation_ConvLSTM.py, which each carried their own copy.

NaN policy: a cell is scored only where every operand is finite. When several
models are compared, build one mask with `valid_mask(model_a, model_b, y_true)`
and pass it to every metric, so all models are reduced over the same
denominator, and report the retained fraction with `mask_coverage`. The same
policy governs `basin_wide_mean_twsa`, so the gridded table and the basin table
score identical data.
"""

import numpy as np
import pandas as pd
import torch
import torchmetrics
import xarray as xr

# GRACE climatology over 2002-04..2008-02, used on GRACE in place of the training climatology
GRACE_CLIMATOLOGY_PATH = 'data/processed/GRACE_climatology_pre2008.nc'


def valid_mask(*tensors):
    """Boolean mask of the cells that are finite in every given tensor."""
    mask = torch.isfinite(tensors[0])
    for tensor in tensors[1:]:
        mask = mask & torch.isfinite(tensor)
    return mask


def mask_coverage(mask, dim=1):
    """Fraction of cells a `valid_mask` keeps, reduced over `dim`."""
    return mask.to(torch.float32).mean(dim=dim)


def _rmse(y_hat, y_true, mask=None, lat_weights=None):
    """Calculate Root Mean Squared Error (RMSE) between two tensors.

    Reduces over dim=1 and over the cells in `mask`, which defaults to the
    cells finite in both inputs.

    `lat_weights` [N], one per cell along dim=1 (cos(lat) for an area-weighted
    RMSE), makes the mean a weighted mean. It is normalised by the weight of the
    kept cells, so the weights need not sum to 1 and, unlike
    `_lat_weighted_rmse`, masked cells do not bias the result low.
    """
    mask = valid_mask(y_hat, y_true) if mask is None else mask
    squared_error = ((y_true - y_hat) ** 2).masked_fill(~mask, 0.0)
    if lat_weights is None:
        return torch.sqrt(squared_error.sum(dim=1) / mask.sum(dim=1))
    weights = lat_weights.reshape(1, -1, *[1] * (y_hat.ndim - 2)) * mask
    return torch.sqrt((squared_error * weights).sum(dim=1) / weights.sum(dim=1))


def _lat_weighted_rmse(y_hat, y_true, lat_weigths, mask=None):
    """Calculate latitude weighted RMSE between two tensors.

    `lat_weigths` [N] is expected to already sum to 1 over the node axis, so
    the weighting is itself the normalisation: the reduction is a sum, not a
    mean. src/models/pl_hmmodel.py pairs the same weights with .sum(dim=[1,2]).
    Masked cells contribute 0, so with incomplete coverage the result is low in
    proportion to the weight mass dropped -- report `mask_coverage` alongside.
    """
    mask = valid_mask(y_hat, y_true) if mask is None else mask
    weighted = (((y_true - y_hat) ** 2) * lat_weigths[None, :, None]).masked_fill(~mask, 0.0)
    return torch.sqrt(weighted.sum(dim=1))


def _corr(y_hat, y_true):
    """Pearson r per lead time, correlated over the prediction windows.

    Inputs are [windows, leads]. The windows are the sample axis: successive
    windows start one month apart, so each value is the correlation of the
    series at a fixed lead -- what a "correlation with GRACE" caption means,
    and the orientation of the Corrlead columns of basin_scores.csv.

    Passing [leads, windows] instead correlates the 6 leads inside one window
    against each other. That is 6 points per estimate, and because Pearson r
    drops each window's own mean it scores a model with the right shape but the
    wrong level a perfect 1.0. There are always far more windows than leads, so
    the transposed call is rejected rather than silently scored.
    """
    y_hat, y_true = torch.as_tensor(y_hat).float(), torch.as_tensor(y_true).float()
    assert y_hat.shape == y_true.shape, f"shape mismatch: {y_hat.shape} vs {y_true.shape}"
    assert y_hat.ndim == 2 and y_hat.shape[0] > y_hat.shape[1], (
        f"expected [windows, leads] with the samples on axis 0, got {tuple(y_hat.shape)}")
    return torchmetrics.PearsonCorrCoef(num_outputs=y_hat.shape[1])(y_hat, y_true)


def _as_window_node_lead(*fields):
    """[windows, nodes, leads] arrays as DataArrays sharing one set of coords."""
    dims = ['window', 'nodes', 'lead']
    coords = {dim: np.arange(size) for dim, size in zip(dims, fields[0].shape)}
    return [xr.DataArray(field, dims=dims, coords=coords) for field in fields]


def gridded_score(score_fn, y_hat, y_true, climatology=None):
    """A `scores.continuous` metric per node and lead, reduced over the windows.

    Inputs are [windows, nodes, leads]. As in `_corr` the windows are the sample
    axis: successive windows start one month apart, so each value scores one
    node's series at one lead. That is what makes the metrics mean what their
    names say -- the NSE denominator is the variance of that series, the KGE
    terms are its bias and variance ratios, and pearsonr is its correlation.

    Reducing over the leads instead scores the 6 leads inside a single window
    against each other: 6 points per estimate, benchmarked against that one
    window's own mean rather than against the record.

    Given a `climatology` [windows, nodes, leads] both fields become anomalies
    first. For pearsonr that is the anomaly correlation (ACC): raw TWSA r is
    dominated by the annual cycle, which the climatology already reproduces, so
    the ACC is the one that shows skill beyond seasonality.
    """
    y_hat, y_true = np.asarray(y_hat), np.asarray(y_true)
    if climatology is not None:
        climatology = np.asarray(climatology)
        y_hat, y_true = y_hat - climatology, y_true - climatology
    arr_hat, arr_true = _as_window_node_lead(y_hat, y_true)
    return score_fn(arr_hat, arr_true, reduce_dims=['window'])


def gridded_skill_score(y_hat, y_true, reference):
    """1 - SSE(y_hat) / SSE(reference) per node and lead, over the windows.

    NSE with an explicit benchmark. `scores`' nse divides by the variance of the
    observed series, i.e. it asks "do you beat the test-period mean of this
    pixel" -- a level the model never saw, so on a trending pixel most of the
    penalty is the offset rather than the dynamics. Passing the training
    climatology as `reference` asks the question a TWSA forecast actually has to
    answer instead: positive means the model beats predicting the seasonal
    cycle, 0 means it matches it, negative means it is worse.

    All three inputs are [windows, nodes, leads]; `reference` is the climatology
    already expanded to the predicted months.
    """
    from scores.continuous import mse

    arr_hat, arr_true, arr_ref = _as_window_node_lead(
        np.asarray(y_hat), np.asarray(y_true), np.asarray(reference))
    return 1 - (mse(arr_hat, arr_true, reduce_dims=['window'])
                / mse(arr_ref, arr_true, reduce_dims=['window']))


def _pixel_r_kge2(y_hat, y_true):
    """Pearson r and KGE″ per node and lead over the windows (axis 0), population sd.

    NaN where a series has zero variance; the caller drops those pixels.
    """
    hat_dev, true_dev = y_hat - y_hat.mean(0), y_true - y_true.mean(0)
    sd_hat, sd_true = np.sqrt((hat_dev ** 2).mean(0)), np.sqrt((true_dev ** 2).mean(0))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (hat_dev * true_dev).mean(0) / (sd_hat * sd_true)
        beta = (y_hat.mean(0) - y_true.mean(0)) / sd_true
        return r, 1 - np.sqrt((r - 1) ** 2 + (sd_hat / sd_true - 1) ** 2 + beta ** 2)


def per_lead_skill_scores(y_hat, y_true, climatology, lat_weights):
    """Area-weighted NSE, KGE″ and r per lead, raw and deseasonalized.

    Inputs are [windows, nodes, leads], the windows being the sample axis as in
    `gridded_score`; `climatology` is the training climatology expanded to the
    predicted months and `lat_weights` [nodes] the cos(lat) cell weights. Returns
    {column: [leads]}, the `clim_*` columns scoring the climatology as forecast.

    NSE is pooled, 1 - sum w*SSE / sum w*SSE_ref over all cells and windows: NSE
    has no lower bound, so a mean of per-pixel NSEs is dominated by a few
    low-variance pixels. The raw NSE benchmarks each pixel's mean over the
    windows, the deseasonalized one the climatology (`gridded_skill_score`).
    r and KGE″ are computed per pixel and averaged with the weights of the
    pixels where they are finite.

    KGE″ (Tang et al., 2021, J. Climate 34(16), 6493-6511,
    https://doi.org/10.1175/JCLI-D-21-0067.1) replaces the standard KGE's bias
    ratio mean_pred/mean_obs by beta″ = (mean_pred - mean_obs)/sd_obs, ideal 0.
    Per-pixel test-period means are close to zero -- raw TWSA has median
    |mean|/sd = 0.14, the anomaly 0.32 -- so the standard ratio is noise.

    A (node, lead) is scored only if all three inputs are finite in every
    window; the weights need not sum to 1.
    """
    y_hat, y_true, climatology = (np.asarray(a, dtype=np.float64) for a in (y_hat, y_true, climatology))
    keep = (np.isfinite(y_hat) & np.isfinite(y_true) & np.isfinite(climatology)).all(axis=0)  # [nodes, leads]
    y_hat, y_true, climatology = (np.where(keep, a, 0.0) for a in (y_hat, y_true, climatology))
    w = np.asarray(lat_weights, dtype=np.float64)[:, None] * keep

    def pooled_nse(pred, reference):
        return 1 - (w * ((pred - y_true) ** 2).sum(0)).sum(0) / (w * ((reference - y_true) ** 2).sum(0)).sum(0)

    def weighted_mean(score):
        ok = np.isfinite(score)
        return (w * np.where(ok, score, 0.0)).sum(0) / (w * ok).sum(0)

    r, kge2 = _pixel_r_kge2(y_hat, y_true)
    r_anom, kge2_anom = _pixel_r_kge2(y_hat - climatology, y_true - climatology)
    r_clim, kge2_clim = _pixel_r_kge2(climatology, y_true)
    mean_true = y_true.mean(0)
    return {'nse_latw': pooled_nse(y_hat, mean_true), 'nse_deseas_latw': pooled_nse(y_hat, climatology),
            'kge2_latw': weighted_mean(kge2), 'kge2_deseas_latw': weighted_mean(kge2_anom),
            'corr_latw': weighted_mean(r), 'corr_deseas_latw': weighted_mean(r_anom),
            'clim_nse_latw': pooled_nse(climatology, mean_true), 'clim_kge2_latw': weighted_mean(kge2_clim),
            'clim_corr_latw': weighted_mean(r_clim)}


def nse(y_hat, y_true):
    """Calculate Nash-Sutcliffe Efficiency (NSE) with long term mean (climatology) or ground truth.
    y_hat: shape [T (time steps), N (num nodes)]
    y_true: shape [T, N]"""
    numerator = torch.sum((y_true - y_hat) ** 2, dim=0)
    denominator = torch.sum((y_true - torch.mean(y_true, dim=0, keepdim=True)) ** 2, dim=0)
    return 1 - numerator / (denominator + 1e-8)


def recover_month_index(month_sin, month_cos):
    # shape: [B, T]
    angle = torch.atan2(month_sin, month_cos)  # range: [-π, π]
    month_float = (angle % (2 * np.pi)) * 12 / (2 * np.pi)  # range: [0, 12)
    month_index = month_float.long()  # round down to nearest integer (0–11)
    return month_index


def compute_climatology(data: xr.DataArray, lsm=None):
    """Compute climatology from the dataset.

    Without `lsm` the gridded climatology [12, H, W] is returned, which is what
    the ConvLSTM evaluation works on. With a land/sea mask the land cells are
    selected and flattened to [12, num land nodes].
    """
    climatology = data.groupby('time.month').mean(dim='time').values  # [12, N]
    if lsm is None:
        return climatology  # [12, H, W]
    land_clim = climatology[lsm[:12]].reshape(12, -1)

    return land_clim  # [12, Num land nodes]


def load_grace_climatology():
    """C_grace [month, lat, lon] in cm, from scripts/preprocess_data/05_grace_climatology.py."""
    return xr.open_dataarray(GRACE_CLIMATOLOGY_PATH).load()


def get_climatology(twsa_sequence_data, config, climatology=None):
    # the given climatology (default: the datamodule's raw training C) of the land nodes, [12, N_land] in cm
    if climatology is None:
        climatology = twsa_sequence_data.twsa_climatology
    lsm0 = twsa_sequence_data.land_ocean_mask.isel(time=0).values
    return climatology.values[:, lsm0].astype(np.float32)


def get_months_output_sequences(months_input_sequences, output_seq_len):
    months_output_sequences = [(months_input_sequences[:,:, -1] +i +1)%12 for i in range(output_seq_len)]
    months_output_sequences = torch.stack([(months_input_sequences[:,:, -1] +i +1)%12 for i in range(output_seq_len)])  # [horizon, Total_B, N]
    months_output_sequences = months_output_sequences.permute(1,2,0)  # [Total_B, N, horizon]
    return months_output_sequences


def basin_grouping(datamodule):
    """Node -> basin grouping. Constant over a run, so it is computed once.

    Returns the sorted basin ids, the node order that makes each basin one
    contiguous segment, and the segment starts np.add.reduceat needs.
    """
    antarctica_mask = (~datamodule.get_antarctica_landmask())
    land2basin = np.asarray(datamodule.data_pred.land_to_basin_ids)[antarctica_mask]
    basin_ids, group_index = np.unique(land2basin, return_inverse=True)
    order = np.argsort(group_index, kind='stable')
    starts = np.searchsorted(group_index[order], np.arange(len(basin_ids)))
    return basin_ids, order, starts


def _grouped_nanmean(values, grouping, weights=None):
    """Mean of `values` [..., N] over the last axis, grouped by basin.

    Sorting the nodes by basin makes every basin a contiguous segment, so a
    single np.add.reduceat reduces all the leading indices at once -- in place
    of one pandas groupby per (window, lead). NaN cells are excluded, which is
    what the groupby did and what the mask policy in `_rmse` does.

    With `weights` [N] (cos(lat) for an area-weighted basin mean) each cell
    counts in proportion to its weight, and a NaN cell drops its weight.
    """
    basin_ids, order, starts = grouping
    values = np.asarray(values)
    flat = values.reshape(-1, values.shape[-1])[:, order]
    finite = np.isfinite(flat)
    w = np.ones(flat.shape[1]) if weights is None else np.asarray(weights, dtype=np.float64)[order]
    totals = np.add.reduceat(np.where(finite, flat * w, 0.0), starts, axis=1)
    counts = np.add.reduceat(finite * w, starts, axis=1)  # weight of the finite cells, 1 each by default
    with np.errstate(invalid='ignore'):
        means = totals / counts
    return means.reshape(values.shape[:-1] + (len(basin_ids),)).astype(values.dtype)


def basin_wide_mean_twsa(y_hat, y_true, climatology, datamodule, grouping=None, lat_weights=None):
    """Calculate the basin wide mean TWSA predictions/ground truth per each time step.
    Should be 238 (includes 1 dummy basin for hawaii) basins in total.
    Args:
        y_hat: predicted values, shape [T, N]
        y_true: true values, shape [T, N]
        datamodule: datamodule containing the land_ocean_mask
        grouping: cached `basin_grouping(datamodule)`, recomputed if omitted
        lat_weights: [N] cell weights (cos(lat)) for area-weighted basin means,
            plain means if omitted
    returns:
        pred_basin_means: predicted basin means, shape [T, N_basins]
        true_basin_means: true basin means, shape [T, N_basins]
    """
    grouping = basin_grouping(datamodule) if grouping is None else grouping
    stacked = np.stack([np.asarray(y_hat), np.asarray(y_true), np.asarray(climatology)])  # [3, T, N]
    pred_basin_means, true_basin_means, climatology_mean = _grouped_nanmean(stacked, grouping, lat_weights)

    return pred_basin_means, true_basin_means, climatology_mean, pd.Index(grouping[0], name='basin')


def get_basin_wide_mean_twsa(twsa_pred, twsa_true, climatology, months_output_sequences, datamodule,
                             lat_weights=None):
    # calculate basin-wide mean TWSA for each prediction sequence
    pred_basin_mean = []
    true_basin_mean = []
    basin_mean_climatology = []
    pred_month = []
    grouping = basin_grouping(datamodule)

    for (pred, true, months) in zip(twsa_pred, twsa_true, months_output_sequences):
        pred, true, basin_clim, basin_ids = basin_wide_mean_twsa(pred.permute(1,0),
                                                                  true.permute(1,0),
                                                                  climatology[months[0]],
                                                                   datamodule,
                                                                   grouping=grouping,
                                                                   lat_weights=lat_weights)
        pred_basin_mean.append(pred)
        true_basin_mean.append(true)
        basin_mean_climatology.append(basin_clim)
        pred_month.append(months[0])
    pred_basin_mean = np.stack(pred_basin_mean)  # [B, T (horizon), N_basins]
    true_basin_mean = np.stack(true_basin_mean)  # [B, T(horizon), N_basins]
    basin_mean_climatology = np.stack(basin_mean_climatology)  # [B, T(horizon), N_basins]
    pred_months = np.stack(pred_month)  # [B, T(horizon)]

    return pred_basin_mean, true_basin_mean, basin_mean_climatology, pred_months, basin_ids
