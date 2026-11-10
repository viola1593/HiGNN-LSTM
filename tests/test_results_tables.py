"""Helpers of the N1 results tables (scripts/results_tables.py)."""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import results_tables as rt


def test_models_are_the_reported_studies():
    assert [model for model, _ in rt.MODELS] == ["HiGNN-LSTM", "No MP", "No ocean nodes", "No ERA5", "ConvLSTM"]


def test_sample_starts_skip_windows_across_a_gap():
    times = pd.date_range("2000-01-01", periods=12, freq="MS")
    times = times.delete(6).values  # 2000-07 missing
    # seq 2, pred 2: a window [t-2, t+2) must span 4 consecutive months
    starts = rt.sample_starts(times, seq_len=2, pred_len=2)
    np.testing.assert_array_equal(starts, [2, 3, 4, 8, 9])


def test_anomaly_persistence_keeps_the_anomaly_of_the_month_before_the_start():
    months = np.arange(6) % 12  # Jan..Jun
    clim = np.arange(12, dtype=float)[:, None] * np.array([1.0, 10.0])  # [12, 2]
    field = clim[months] + np.array([[0, 0], [0, 0], [2.0, -1.0], [9, 9], [9, 9], [9, 9]])

    pred = rt.anomaly_persistence(field, clim, months, np.array([3]), 2)

    # start t=3 (Apr): anomaly at t-1 = (2, -1), added to the climatology of Apr and May
    assert pred.shape == (1, 2, 2)  # [windows, nodes, leads]
    np.testing.assert_allclose(pred[0], [[3 + 2.0, 4 + 2.0], [30 - 1.0, 40 - 1.0]])


@pytest.fixture
def train_and_grace_clim(tmp_path, monkeypatch):
    """Two years of training twsa on a 2x3 grid (one NaN cell) and a C_grace file on the same grid."""
    lat, lon = np.array([10.0, -10.0]), np.array([0.0, 1.0, 2.0])
    time = pd.date_range("2000-01-01", periods=24, freq="MS")
    values = np.arange(24 * 6, dtype=np.float32).reshape(24, 2, 3)
    values[:, 0, 0] = np.nan
    train = xr.DataArray(values, coords={"time": time, "lat": lat, "lon": lon}, dims=("time", "lat", "lon"))
    c_grace = xr.DataArray(np.random.default_rng(0).normal(size=(12, 2, 3)).astype(np.float32),
                           coords={"month": np.arange(1, 13), "lat": lat, "lon": lon}, dims=("month", "lat", "lon"))
    c_grace.to_dataset(name="twsa").to_netcdf(tmp_path / "c_grace.nc")
    monkeypatch.setattr(rt, "GRACE_CLIM_PATH", str(tmp_path / "c_grace.nc"))
    return train, c_grace


@pytest.mark.parametrize("dataset", ["val", "test", "grace"])
def test_climatology_is_the_training_climatology_nan_to_zero(train_and_grace_clim, dataset):
    train, _ = train_and_grace_clim

    clim = rt.climatology(dataset, train)

    expected = (train.values[:12] + train.values[12:]) / 2  # Jan..Dec of the two years
    expected[:, 0, 0] = 0.0
    assert clim.shape == (12, 2, 3) and clim.dtype == np.float32
    np.testing.assert_allclose(clim, expected)


def test_climatology_is_c_grace_on_grace_graceclim(train_and_grace_clim):
    train, c_grace = train_and_grace_clim

    clim = rt.climatology("grace_graceclim", train)

    np.testing.assert_array_equal(clim, c_grace.values)


def test_seed_summary_is_mean_and_sample_sd_per_lead():
    seeds = [pd.DataFrame({"rmse_latw": [1.0, 2.0]}), pd.DataFrame({"rmse_latw": [3.0, 2.0]})]

    mean, sd = rt.seed_summary(seeds, "rmse_latw")

    np.testing.assert_allclose(mean, [2.0, 2.0])
    np.testing.assert_allclose(sd, [np.sqrt(2.0), 0.0])


def test_latex_table_rows():
    rows = [("HiGNN-LSTM", [("RMSE [cm]", np.array([1.0, 2.0]), np.array([0.01, 0.002]))]),
            ("Persistence", [("RMSE [cm]", np.array([1.5, 2.5]), None)])]

    tex = rt.latex_table(rows, caption="cap", label="tab:x")

    assert "Model & Metric & L1 & L2 \\\\" in tex
    assert "HiGNN-LSTM & RMSE [cm] & $1.000 \\pm 0.010$ & $2.000 \\pm 0.002$ \\\\" in tex
    assert "Persistence & RMSE [cm] & $1.500$ & $2.500$ \\\\" in tex
    assert "\\label{tab:x}" in tex


def test_welch_is_the_unequal_variance_t_test_of_the_seed_means():
    full = np.array([2.0, 2.1, 1.9, 2.0])
    abl = np.array([2.3, 2.2, 2.6, 2.5, 2.4])

    d, lo, hi, p = rt.welch(abl, full)

    # Welch by hand: SE from the two sample variances, Welch-Satterthwaite degrees of freedom
    va, vb = abl.var(ddof=1) / 5, full.var(ddof=1) / 4
    se, df = np.sqrt(va + vb), (va + vb) ** 2 / (va ** 2 / 4 + vb ** 2 / 3)
    q = stats.t.ppf(0.975, df)
    assert d == pytest.approx(abl.mean() - full.mean())
    assert p == pytest.approx(2 * stats.t.sf(d / se, df))
    assert (lo, hi) == pytest.approx((d - q * se, d + q * se))


def test_ablation_rows_give_differences_in_cm():
    full = {"val": np.array([2.0, 2.1, 1.9, 2.0]), "test": np.array([1.0, 1.1])}
    abl = {"val": np.array([2.3, 2.2, 2.6, 2.5, 2.4]), "test": np.array([1.2, 1.3])}

    rows = rt.ablation_rows({"Full": full, "Abl": abl})

    assert [m for m, _ in rows] == ["Full", "Abl"]
    mean, sd, s = rows[0][1][0]
    assert (mean, sd, s) == (pytest.approx(2.0), pytest.approx(full["val"].std(ddof=1)), None)
    d, lo, hi, p = rt.welch(abl["test"], full["test"])
    np.testing.assert_allclose(rows[1][1][1][2], [d, lo, hi, p])


ROWS = [("Full", [(2.0, 0.01, None)]), ("Abl", [(2.01, 0.02, (0.012, 0.004, 0.02, 0.0004))])]


def test_ablation_markdown_rows():
    md = rt.ablation_markdown(ROWS, ["Val"])

    assert "| Variant | Val RMSE [cm] | Val ΔRMSE [cm] (95 % CI) | Val p |" in md
    assert "| Full | 2.000 ± 0.010 | – | – |" in md
    assert "| Abl | 2.010 ± 0.020 | +0.012 (+0.004, +0.020) | <0.001 |" in md


def test_ablation_latex_rows():
    tex = rt.ablation_latex(ROWS, ["Val"], caption="cap", label="tab:abl")

    assert "Variant & RMSE [cm] & $\\Delta$RMSE [cm] (95\\,\\% CI) & $p$ \\\\" in tex
    assert "Full & $2.000 \\pm 0.010$ & -- & -- \\\\" in tex
    assert "Abl & $2.010 \\pm 0.020$ & $+0.012$ ($+0.004$, $+0.020$) & $<0.001$ \\\\" in tex
    assert "\\label{tab:abl}" in tex


# per-lead rows: ablation_rows with one key per lead
LEAD_ROWS = [("Full", [(2.0, 0.01, None), (3.0, 0.02, None)]),
             ("Abl", [(2.01, 0.02, (0.012, 0.004, 0.02, 0.0004)), (3.0, 0.0, (-0.003, -0.01, 0.004, 0.5))])]


def test_ablation_lead_markdown_has_leads_as_columns():
    md = rt.ablation_lead_markdown(LEAD_ROWS)

    assert "| Model | Metric | L1 | L2 |" in md
    assert "| Full | RMSE [cm] | 2.000 ± 0.010 | 3.000 ± 0.020 |" in md
    assert "| Abl | ΔRMSE [cm] | +0.012 | -0.003 |" in md
    assert "|  | 95 % CI | (+0.004, +0.020) | (-0.010, +0.004) |" in md
    assert "|  | p | <0.001 | 0.500 |" in md


def test_ablation_lead_latex_has_leads_as_columns():
    tex = rt.ablation_lead_latex(LEAD_ROWS, caption="cap", label="tab:lead")

    assert "Model & Metric & L1 & L2 \\\\" in tex
    assert "Full & RMSE [cm] & $2.000 \\pm 0.010$ & $3.000 \\pm 0.020$ \\\\" in tex
    assert "\\multirow{3}{*}{Abl} & $\\Delta$RMSE [cm] & $+0.012$ & $-0.003$ \\\\" in tex
    assert " & 95\\,\\% CI & ($+0.004$, $+0.020$) & ($-0.010$, $+0.004$) \\\\" in tex
    assert " & $p$ & $<0.001$ & $0.500$ \\\\" in tex
    assert "\\label{tab:lead}" in tex
