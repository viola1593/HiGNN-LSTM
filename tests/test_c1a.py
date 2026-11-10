"""Batch C1a: evaluation_HiGNN.py -- the RMSE-drift gap (#8), the per-basin r (#12)
and the eval dir prefix (#25)."""

import pathlib
import re
import sys

import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch
import torchmetrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import evaluation_HiGNN
from utils.eval_metrics import _corr
from utils.utils import plot_rmse_broken_axis


# ---------------------------------------------------------------- #12: per-basin r


def _basin_windows(num_windows=40, num_leads=6):
    """Sliding windows over one sinusoidal basin series, plus a per-lead offset, and a
    phase-shifted climatology so the anomalies are not zero.

    A constant offset per lead leaves the across-window correlation of every lead
    untouched (r = 1 for each), but wrecks the across-lead correlation inside a
    window -- so the two orientations are far apart on this input.
    """
    months = np.arange(num_windows + num_leads)
    series = np.sin(2 * np.pi * months / 12)
    seasonal = np.cos(2 * np.pi * months / 12)
    true = np.stack([series[b:b + num_leads] for b in range(num_windows)])
    clim = np.stack([seasonal[b:b + num_leads] for b in range(num_windows)])
    pred = true + np.arange(num_leads) * 5.0
    return pred[:, :, None], true[:, :, None], clim[:, :, None]  # [B, T, 1], as the basin slices are


def _title_r(name, pred, true, clim):
    fig, ax = plt.subplots()
    evaluation_HiGNN.plot_basintwsa_timesteps(name, 1, pred, true, clim, ax=ax)
    title = ax.get_title()
    plt.close(fig)
    return float(re.search(r"\$r_\{anom\}\$=(-?\d+\.\d+)", title).group(1))


def test_basin_panel_r_is_the_lead_wise_correlation():
    """The panel r must be the mean over leads of the anomaly correlation across windows."""
    pred, true, clim = _basin_windows()
    expected = _corr(pred[:, :, 0] - clim[:, :, 0], true[:, :, 0] - clim[:, :, 0]).mean().item()

    assert expected == pytest.approx(1.0, abs=1e-3)
    assert _title_r("Amazon", pred, true, clim) == pytest.approx(expected, abs=5e-3)


def test_basin_panel_r_is_not_the_within_window_correlation():
    """Guard: the old orientation (6 lead times as the samples) differs on this input."""
    pred, true, clim = _basin_windows()
    within_window = torchmetrics.PearsonCorrCoef(num_outputs=pred.shape[0])(
        torch.Tensor(pred - clim).squeeze().permute(1, 0),
        torch.Tensor(true - clim).squeeze().permute(1, 0)).mean().item()

    assert abs(within_window - 1.0) > 0.1
    assert abs(_title_r("Amazon", pred, true, clim) - within_window) > 0.1


# ------------------------------------------------------------------- #8: the gap


def _rmse_series(dates, num_leads=6):
    return np.arange(num_leads * len(dates), dtype=float).reshape(num_leads, len(dates))


def _lead_points(axes):
    """Number of points drawn for lead 1, summed over the panels."""
    return sum(len(line.get_xdata()) for ax in axes for line in ax.lines
               if line.get_label() == "Lead 1")


CONTIGUOUS = pd.date_range("2015-01-01", periods=40, freq="MS")
WITH_GAP = pd.date_range("2015-01-01", periods=30, freq="MS").append(
    pd.date_range("2018-05-01", periods=20, freq="MS"))


def test_find_month_gap_reports_none_on_a_contiguous_series():
    assert evaluation_HiGNN.find_month_gap(CONTIGUOUS.values) == (None, None)


def test_find_month_gap_finds_the_grace_gap():
    assert evaluation_HiGNN.find_month_gap(WITH_GAP.values) == (29, 30)


def test_find_month_gap_takes_the_widest_gap():
    dates = (pd.date_range("2015-01-01", periods=5, freq="MS")
             .append(pd.date_range("2015-07-01", periods=5, freq="MS"))
             .append(pd.date_range("2018-05-01", periods=5, freq="MS")))
    assert evaluation_HiGNN.find_month_gap(dates.values) == (9, 10)


def test_contiguous_series_is_one_panel_with_no_gap_annotation(tmp_path):
    fig, axes = plot_rmse_broken_axis(dates=CONTIGUOUS, rmse=_rmse_series(CONTIGUOUS),
                                      gap_start_id=None, gap_end_id=None,
                                      savepath=str(tmp_path / "rmse.png"))
    try:
        assert len(axes) == 1
        assert _lead_points(axes) == len(CONTIGUOUS)
        assert not any("data gap" in text.get_text() for text in fig.texts)
    finally:
        plt.close(fig)


def test_a_real_gap_breaks_the_axis_at_the_gap_and_drops_no_point(tmp_path):
    start, end = evaluation_HiGNN.find_month_gap(WITH_GAP.values)
    fig, axes = plot_rmse_broken_axis(dates=WITH_GAP, rmse=_rmse_series(WITH_GAP),
                                      gap_start_id=start, gap_end_id=end,
                                      savepath=str(tmp_path / "rmse.png"))
    try:
        assert len(axes) == 2
        assert _lead_points(axes) == len(WITH_GAP)
        annotation = " ".join(text.get_text() for text in fig.texts)
        assert "data gap" in annotation
        assert "2017-06" in annotation and "2018-05" in annotation
    finally:
        plt.close(fig)
