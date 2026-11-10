"""Per-basin input removal (scripts/basin_latent_removal.py)."""
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import basin_latent_removal as blr
from utils.eval_metrics import basin_grouping

REMOVALS = ["dim1", "dim2", "dim3", "dim4", "twsa", "lead"]


def toy_grouping():
    """3 cells: cells 0 and 2 in basin 10, cell 1 in basin 20."""
    dm = SimpleNamespace(get_antarctica_landmask=lambda: np.zeros(3, bool),
                         data_pred=SimpleNamespace(land_to_basin_ids=np.array([10, 20, 10])))
    return basin_grouping(dm)


def test_basin_rmse_scores_the_area_weighted_basin_mean_and_skips_missing_truth():
    # [windows 2, cells 3, leads 1]
    pred = np.array([[[1.0], [5.0], [3.0]], [[2.0], [0.0], [4.0]]])
    true = np.array([[[0.0], [4.0], [0.0]], [[0.0], [1.0], [np.nan]]])
    w = np.array([1.0, 1.0, 3.0])

    rmse = blr.basin_rmse(pred, true, toy_grouping(), w)

    assert rmse.shape == (1, 2)  # [leads, basins]
    # basin 10: window 0 pred (1 + 9) / 4 = 2.5, truth 0; window 1 cell 2 has no truth: pred 2, truth 0
    assert rmse[0, 0] == pytest.approx(np.sqrt((2.5 ** 2 + 2.0 ** 2) / 2))
    # basin 20: errors 1 and -1
    assert rmse[0, 1] == pytest.approx(1.0)


def long_scores(rmse):
    """`rmse` {removal: [leads, basins]} -> long rows as the script caches them."""
    return pd.DataFrame([dict(seed="45", removal=r, basin=b, lead=lead, rmse=v[lead, k])
                         for r, v in rmse.items() for lead in range(v.shape[0])
                         for k, b in enumerate([10, 20, 30])])


def test_lead_mean_deltas_per_basin_in_label_order_and_drop_missing_basins():
    rng = np.random.default_rng(0)
    rmse = {r: 2 + rng.random((2, 3)) for r in ["none", *REMOVALS[::-1]]}
    rmse["dim3"][1, 2] = np.nan

    deltas = blr.lead_mean_deltas(long_scores(rmse))

    assert list(deltas.columns) == REMOVALS
    assert list(deltas.index) == [10, 20]
    for r in REMOVALS:
        for k, b in enumerate([10, 20]):
            assert deltas.loc[b, r] == pytest.approx((rmse[r][:, k] - rmse["none"][:, k]).mean())


def test_spearman_matches_scipy():
    rng = np.random.default_rng(1)
    deltas = pd.DataFrame(rng.normal(size=(20, 6)), columns=REMOVALS)

    corr = blr.spearman(deltas)

    assert list(corr.index) == REMOVALS and list(corr.columns) == REMOVALS
    assert np.allclose(corr.to_numpy(), spearmanr(deltas.to_numpy()).statistic)


def test_markdown_counts_basins_that_get_better():
    deltas = pd.DataFrame({r: [0.1, 0.2, 0.3] for r in REMOVALS}, index=[10, 20, 30])
    deltas["dim2"] = [-0.1, -0.2, 0.3]
    global_deltas = pd.Series({r: 0.05 for r in REMOVALS})

    md = blr.markdown(deltas, global_deltas, "45")

    assert "| Dim 2 | +0.050 | -0.100 | 2 / 3 |" in md  # global Δ, median basin Δ, basins with Δ < 0
    assert "| Dim 1 | +0.050 | +0.200 | 0 / 3 |" in md
    assert "seed 45" in md
