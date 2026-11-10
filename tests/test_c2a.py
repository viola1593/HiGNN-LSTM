"""Batch C2a: compare_models.py -- missing BasinPlots dir (#2), climatology crop (#5),
checkpoint choice (#7), `dataset` shadowing (#27), and the seq_len/delay guard."""

import os
import pathlib
import sys
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

matplotlib.use("Agg")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import compare_models


# ---------------------------------------------------------------- #2: output dirs


def test_create_experiment_dir_creates_basinplots(tmp_path):
    config = {"experiment": {"save_dir": str(tmp_path / "run")}}
    updated = compare_models.create_experiment_dir(config, dataset="test")
    assert os.path.isdir(os.path.join(updated["experiment"]["save_dir"], "BasinPlots"))


# ---------------------------------------------------------------- #7: checkpoint choice


def _touch(path, mtime):
    path.touch()
    os.utime(path, (mtime, mtime))


def _loaded_ckpt(monkeypatch, run_dir):
    loaded = []

    class FakeHMModule:
        @staticmethod
        def load_from_checkpoint(path, **kwargs):
            loaded.append(path)
            return SimpleNamespace(model_summary=lambda: None)

    monkeypatch.setattr(compare_models, "HMModule", FakeHMModule)
    seq = SimpleNamespace(num_features=1, num_ocean_features=1, use_ocean_nodes=False,
                          input_seq_len=12, target_seq_len=6)
    compare_models.build_model_from_checkpoint(str(run_dir), {"model": {}}, 0, seq)
    return os.path.basename(loaded[0])


def test_best_ckpt_is_chosen_on_a_resumed_run(tmp_path, monkeypatch):
    """After a resume the best-val ckpt can be newer than last.ckpt; mtime must not decide."""
    _touch(tmp_path / "last.ckpt", 1_000)
    _touch(tmp_path / "epoch=41-step=420.ckpt", 2_000)
    assert _loaded_ckpt(monkeypatch, tmp_path) == "epoch=41-step=420.ckpt"


def test_ambiguous_ckpts_are_rejected(tmp_path, monkeypatch):
    _touch(tmp_path / "last.ckpt", 3_000)
    _touch(tmp_path / "epoch=1-step=10.ckpt", 1_000)
    _touch(tmp_path / "epoch=9-step=90.ckpt", 2_000)
    with pytest.raises(AssertionError):
        _loaded_ckpt(monkeypatch, tmp_path)


# ---------------------------------------------------------------- evaluate() harness

T_OUT, T_IN, START = 6, 12, 17  # leads, input months, first window predicting Nov 2019
B = START + 10                   # windows before the crop; 10 survive it (> T_OUT for _corr)
N_ALL, H, W = 20, 4, 5           # land nodes, grid
FIRST_TARGET = np.datetime64("2018-06", "M")


def _month0(dates):
    """Month index with January = 0, as month_to_features encodes it."""
    return dates.astype("datetime64[M]").astype(int) % 12


def _harness(monkeypatch, tmp_path, datamodule_cfg=None):
    rng = np.random.default_rng(0)

    lat = np.linspace(-50, 60, N_ALL)
    lat[:2] = -70  # two Antarctica nodes, masked out by evaluate
    land = np.stack([np.linspace(-170, 170, N_ALL), lat], axis=1)
    antarctica = land[:, 1] < -60

    lsm = np.zeros((T_OUT, H, W), dtype=bool)
    lsm.reshape(T_OUT, -1)[:, :N_ALL] = True

    test_basins = list(compare_models.TEST_BASIN_IDS.values())
    land2basin = pd.Series([test_basins[i % len(test_basins)] for i in range(N_ALL)])

    # window w: inputs are months FIRST_TARGET + w - 12 .. -1, targets FIRST_TARGET + w + l
    times = FIRST_TARGET - T_IN + np.arange(T_IN + B + T_OUT)
    samples = np.arange(T_IN, T_IN + B)
    target_months = (_month0(times[samples])[:, None] + np.arange(T_OUT)) % 12  # [B, T]

    climatology = rng.standard_normal((12, N_ALL)).astype("float32") * 10
    # truth *is* the climatology of the right month, so the clim RMSE must be 0
    true = np.transpose(climatology[target_months], (0, 2, 1))  # [B, N, T]
    pred = true + 2.0

    input_months = _month0(times[samples][:, None] + np.arange(-T_IN, 0))  # [B, T_IN]
    angle = 2 * np.pi * np.repeat(input_months, N_ALL, axis=0) / 12        # [B*N, T_IN]
    x = torch.tensor(np.stack([np.cos(angle), np.sin(angle)], axis=-1), dtype=torch.float32)
    batch = (torch.tensor(pred).reshape(-1, T_OUT), torch.tensor(true).reshape(-1, T_OUT),
             {"grid": SimpleNamespace(x=x)})

    # Li et al.: one variable per initialisation (YYYY-MM, the month before the first
    # prediction), the cropped windows plus 6 surplus ones
    kept = true[START:][:, ~antarctica]
    fcast_vars = {}
    for i in range(B - START + T_OUT):
        field = np.full((T_OUT, H, W), np.nan, dtype="float32")
        if i < B - START:
            full = np.zeros((N_ALL, T_OUT), dtype="float32")
            full[~antarctica] = kept[i] + 1.0
            field.reshape(T_OUT, -1)[:, :N_ALL] = full.T
        fcast_vars[str(FIRST_TARGET + START - 1 + i)] = (("lead", "lat", "lon"), field)
    ds_fcast = xr.Dataset(fcast_vars)

    data_pred = SimpleNamespace(
        time_period=(str(times[0]), str(times[-1])), samples=samples,
        ds_timeslice=SimpleNamespace(time=SimpleNamespace(values=times.astype("datetime64[ns]"))),
        num_land_nodes=lambda: N_ALL, land_to_basin_ids=land2basin)
    datamodule = SimpleNamespace(
        setup=lambda stage: None, data_pred=data_pred, land_ocean_mask=lsm,
        data_train=SimpleNamespace(land=[land]), get_antarctica_landmask=lambda: antarctica)

    dm_cfg = {"input_seq_len": T_IN, "output_seq_len": T_OUT, "delay": 0}
    dm_cfg.update(datamodule_cfg or {})
    config = {"experiment": {"save_dir": str(tmp_path)}, "pl": {}, "datamodule": dm_cfg}

    monkeypatch.setattr(compare_models, "read_config", lambda path: config)
    monkeypatch.setattr(compare_models, "generate_trainer", lambda config: SimpleNamespace())
    monkeypatch.setattr(compare_models, "TWSA_HeteroGraphDatamodule", lambda **kw: datamodule)
    monkeypatch.setattr(compare_models, "get_climatology", lambda seq, config: climatology)
    monkeypatch.setattr(compare_models, "get_predictions", lambda *a, **kw: [batch])
    monkeypatch.setattr(compare_models.xr, "open_dataset", lambda path: ds_fcast)


def _run(monkeypatch, tmp_path, **datamodule_cfg):
    _harness(monkeypatch, tmp_path, datamodule_cfg)
    compare_models.evaluate(str(tmp_path), gpu=0, FCast_file_path="unused.nc")
    return pd.read_csv(tmp_path / "compare_to_fcast" / "fcastper_timestep_scores.csv",
                       index_col="Timestep")


# ---------------------------------------------------------------- #27: `dataset` shadowing


def test_evaluate_reads_every_forecast_variable(monkeypatch, tmp_path):
    """fcast = true + 1 on every kept cell, so its RMSE is exactly 1 at every lead."""
    scores = _run(monkeypatch, tmp_path)
    np.testing.assert_allclose(scores["rmse"], 1.0, atol=1e-5)
    np.testing.assert_allclose(scores["gnn_rmse"], 2.0, atol=1e-5)


# ---------------------------------------------------------------- #5: climatology crop


def test_climatology_is_aligned_with_the_cropped_windows(monkeypatch, tmp_path):
    """Truth equals the climatology of its own months, so clim RMSE must be 0."""
    scores = _run(monkeypatch, tmp_path)
    np.testing.assert_allclose(scores["clim_rmse"], 0.0, atol=1e-5)


# ---------------------------------------------------------------- seq_len / delay guard


@pytest.mark.parametrize("cfg, match", [({"output_seq_len": 3}, "output_seq_len"),
                                        ({"input_seq_len": 24}, "input_seq_len"),
                                        ({"delay": 1}, "delay")])
def test_evaluate_rejects_configs_it_hardcodes_against(monkeypatch, tmp_path, cfg, match):
    with pytest.raises(AssertionError, match=match):
        _run(monkeypatch, tmp_path, **cfg)


# ---------------------------------------------------------------- cos(lat) weighting


def _harness_lat():
    """The node latitudes _harness builds; evaluate keeps the ones >= -60."""
    lat = np.linspace(-50, 60, N_ALL)
    lat[:2] = -70
    return lat


def test_lat_weighted_columns_are_written(monkeypatch, tmp_path):
    """Constant errors (fcast +1, gnn +2, truth = climatology) are weight-independent."""
    scores = _run(monkeypatch, tmp_path)
    np.testing.assert_allclose(scores["rmse_latw"], 1.0, atol=1e-5)
    np.testing.assert_allclose(scores["gnn_rmse_latw"], 2.0, atol=1e-5)
    np.testing.assert_allclose(scores["clim_rmse_latw"], 0.0, atol=1e-5)


def test_gnn_rmse_latw_weights_the_nodes_by_cos_lat(monkeypatch, tmp_path):
    """GNN error 1 north of 20°N and 0 elsewhere: the plain RMSE counts every node
    alike, the weighted one shrinks the high-latitude nodes by cos(lat)."""
    _harness(monkeypatch, tmp_path)
    pred, true, x = compare_models.get_predictions()[0]
    lat = _harness_lat()
    error = (lat > 20).astype("float32")
    pred = true + torch.tensor(np.tile(error, B))[:, None]  # rows are window-major [B*N, T]
    monkeypatch.setattr(compare_models, "get_predictions", lambda *a, **kw: [(pred, true, x)])
    compare_models.evaluate(str(tmp_path), gpu=0, FCast_file_path="unused.nc")
    scores = pd.read_csv(tmp_path / "compare_to_fcast" / "fcastper_timestep_scores.csv",
                         index_col="Timestep")

    kept = lat >= -60
    w, e = np.cos(np.deg2rad(lat[kept])), error[kept]
    np.testing.assert_allclose(scores["gnn_rmse_latw"], np.sqrt((w * e ** 2).sum() / w.sum()), atol=1e-5)
    np.testing.assert_allclose(scores["gnn_rmse"], np.sqrt((e ** 2).mean()), atol=1e-5)


def test_basin_means_are_area_weighted(monkeypatch, tmp_path):
    """Both basin-mean calls (GNN and Li et al.) weight the kept nodes by cos(lat)."""
    _harness(monkeypatch, tmp_path)
    passed = []
    real = compare_models.get_basin_wide_mean_twsa

    def spy(*args, **kwargs):
        passed.append(kwargs.get("lat_weights"))
        return real(*args, **kwargs)

    monkeypatch.setattr(compare_models, "get_basin_wide_mean_twsa", spy)
    compare_models.evaluate(str(tmp_path), gpu=0, FCast_file_path="unused.nc")

    lat = _harness_lat()
    expected = np.cos(np.deg2rad(lat[lat >= -60]))
    assert len(passed) == 2
    for weights in passed:
        weights = np.asarray(weights)
        np.testing.assert_allclose(weights / weights.sum(), expected / expected.sum(), rtol=1e-5)
