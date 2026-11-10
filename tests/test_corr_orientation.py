"""The correlation orientation (#12), in `_corr` and the gridded maps.

Every correlation in the evaluation is a correlation *across the prediction
windows at a fixed lead*: successive windows are successive months, so that is
the correlation of the basin/node time series. Correlating the 6 leads inside a
single window instead scores 6 points against 6 points and, because Pearson r
drops each window's own mean, is blind to any error in the window's level --
`test_within_window_r_is_blind_to_a_level_error` pins that failure down.
"""

import numpy as np
import pytest
import torch
import torchmetrics
import xarray as xr

from scores.continuous import kge, nse
from scores.continuous.correlation import pearsonr
from utils.eval_metrics import _corr, gridded_score, gridded_skill_score


# --------------------------------------------------------------- the _corr contract


def _windows(num_windows=60, num_leads=6, seed=0):
    """Sliding 6-month windows over one sinusoidal series, plus per-lead noise."""
    rng = np.random.default_rng(seed)
    months = np.arange(num_windows + num_leads)
    series = 10 * np.sin(2 * np.pi * months / 12)
    true = np.stack([series[b:b + num_leads] for b in range(num_windows)])
    pred = true + rng.normal(0, 2.0, true.shape)
    return pred, true  # [B, T]


def test_corr_takes_windows_by_leads_and_returns_one_r_per_lead():
    pred, true = _windows()
    out = _corr(pred, true)

    assert out.shape == (6,)
    for lead in range(6):
        expected = np.corrcoef(pred[:, lead], true[:, lead])[0, 1]
        assert out[lead].item() == pytest.approx(expected, abs=1e-4)


def test_corr_derives_the_lead_count_from_the_input():
    """The lead axis is not hardwired to 6 -- output_seq_len is a config value."""
    pred, true = _windows(num_leads=3)
    assert _corr(pred, true).shape == (3,)


def test_corr_rejects_the_transposed_orientation():
    """[leads, windows] must fail loudly rather than silently correlate 6 points."""
    pred, true = _windows()
    with pytest.raises(AssertionError, match="samples"):
        _corr(pred.T, true.T)


def test_within_window_r_is_blind_to_a_level_error():
    """Why the orientation matters: a model with the right shape but a random
    level per window is useless, and only the lead-wise r says so."""
    _, true = _windows()
    rng = np.random.default_rng(1)
    useless = true + rng.normal(0, 15.0, (true.shape[0], 1))  # constant inside a window

    within_window = torchmetrics.PearsonCorrCoef(num_outputs=true.shape[0])(
        torch.Tensor(useless).permute(1, 0), torch.Tensor(true).permute(1, 0)).mean()

    assert within_window.item() == pytest.approx(1.0, abs=1e-3)  # scores it perfect
    assert _corr(useless, true).mean().item() < 0.8              # lead-wise does not


# ----------------------------------------------------------------- the gridded maps


def _gridded(num_windows=60, num_nodes=4, num_leads=6, seed=0):
    """[windows, nodes, leads] fields plus the climatology of the same shape."""
    rng = np.random.default_rng(seed)
    months = np.arange(num_windows + num_leads)
    season = 10 * np.sin(2 * np.pi * months / 12)
    true = np.stack([season[b:b + num_leads] for b in range(num_windows)])
    true = np.repeat(true[:, None, :], num_nodes, axis=1)
    true = true + rng.normal(0, 1.0, true.shape)
    pred = true + rng.normal(0, 2.0, true.shape)
    clim = np.broadcast_to(
        np.stack([season[b:b + num_leads] for b in range(num_windows)])[:, None, :],
        true.shape).copy()
    return pred, true, clim


def test_gridded_corr_reduces_over_the_windows_not_the_leads():
    pred, true, _ = _gridded()
    out = gridded_score(pearsonr, pred, true)

    assert out.dims == ("nodes", "lead")
    assert out.shape == (4, 6)
    for node in range(4):
        for lead in range(6):
            expected = np.corrcoef(pred[:, node, lead], true[:, node, lead])[0, 1]
            assert out.values[node, lead] == pytest.approx(expected, abs=1e-4)


def test_gridded_corr_subtracts_the_climatology_when_given_one():
    pred, true, clim = _gridded()
    out = gridded_score(pearsonr, pred, true, climatology=clim)

    expected = np.corrcoef(pred[:, 0, 0] - clim[:, 0, 0], true[:, 0, 0] - clim[:, 0, 0])[0, 1]
    assert out.values[0, 0] == pytest.approx(expected, abs=1e-4)
    # removing the seasonal cycle leaves the harder signal, so ACC < raw r
    assert out.mean().item() < gridded_score(pearsonr, pred, true).mean().item()


def test_gridded_nse_denominator_is_the_variance_of_the_series():
    """NSE per node and lead, benchmarked against the mean of that node's series
    over the windows -- not against the mean of one 6-month window."""
    pred, true, _ = _gridded()
    out = gridded_score(nse, pred, true)

    assert out.dims == ("nodes", "lead")
    for node, lead in [(0, 0), (2, 5), (3, 1)]:
        p, t = pred[:, node, lead], true[:, node, lead]
        expected = 1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()
        assert out.values[node, lead] == pytest.approx(expected, abs=1e-4)


def test_gridded_kge_matches_the_kge_of_that_single_series():
    pred, true, _ = _gridded()
    out = gridded_score(kge, pred, true)

    node, lead = 1, 3
    series = {"pred": pred[:, node, lead], "true": true[:, node, lead]}
    one = kge(xr.DataArray(series["pred"], dims=["window"]),
              xr.DataArray(series["true"], dims=["window"]), reduce_dims=["window"])
    assert out.values[node, lead] == pytest.approx(one.item(), abs=1e-4)


def test_skill_score_benchmarks_against_the_climatology_not_the_series_mean():
    """NSE with the climatology as the reference: 1 - SSE(model)/SSE(climatology)."""
    pred, true, clim = _gridded()
    out = gridded_skill_score(pred, true, clim)

    assert out.dims == ("nodes", "lead")
    for node, lead in [(0, 0), (2, 5), (3, 1)]:
        p, t, c = pred[:, node, lead], true[:, node, lead], clim[:, node, lead]
        expected = 1 - ((t - p) ** 2).sum() / ((t - c) ** 2).sum()
        assert out.values[node, lead] == pytest.approx(expected, abs=1e-4)


def test_skill_score_is_zero_for_the_climatology_and_one_for_a_perfect_forecast():
    """The two anchors that make the sign readable: >0 beats climatology."""
    pred, true, clim = _gridded()

    assert gridded_skill_score(clim, true, clim).values == pytest.approx(0.0, abs=1e-6)
    assert gridded_skill_score(true, true, clim).values == pytest.approx(1.0, abs=1e-6)


def test_skill_score_differs_from_nse_when_the_series_mean_is_not_the_climatology():
    """The whole point: on a trending pixel the two benchmarks diverge."""
    pred, true, clim = _gridded()
    trend = 0.05 * np.arange(true.shape[0])[:, None, None]
    true, pred = true + trend, pred + trend  # model tracks the trend, clim cannot

    assert (gridded_skill_score(pred, true, clim).mean().item()
            != pytest.approx(gridded_score(nse, pred, true).mean().item(), abs=0.05))


def test_gridded_nse_differs_from_the_within_window_orientation():
    """Guard: reducing the leads instead scores each 6-month segment against its
    own mean, which is a different (and far weaker) benchmark."""
    pred, true, _ = _gridded()
    dims, coords = ["window", "nodes", "lead"], None
    arrs = [xr.DataArray(a, dims=dims, coords=coords) for a in (pred, true)]
    within_window = nse(*arrs, reduce_dims=["lead"]).mean().item()

    assert abs(gridded_score(nse, pred, true).mean().item() - within_window) > 0.1
