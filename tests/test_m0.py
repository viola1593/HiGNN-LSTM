"""M0 -- deduplication of the pure metric helpers into src/utils/eval_metrics.py.

Workflow (the capture has to happen *before* the helpers are moved):

    python tests/test_m0.py capture     # run while the scripts still define them
    pytest tests/test_m0.py             # run after the move

`capture` records the output of every moved helper, for fixed seeded inputs,
from each script that defines it.  The pytest run replays the identical inputs
through src/utils/eval_metrics.py and asserts torch.equal against the recording,
except for the four helpers with a deliberate behaviour change (#30, #14, #6,
#18), which are asserted against their new documented behaviour and printed
old-vs-new.
"""

import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# baseline dir next to the repo (override with M0_BASELINE_DIR)
BASELINE_DIR = pathlib.Path(
    os.environ.get("M0_BASELINE_DIR", ROOT.parent / ".m0-baseline")
)

# node counts of the toy world: 50 land nodes, 10 of which are "antarctica"
N_TOTAL, N_ANT, T, B = 50, 10, 6, 5
N = N_TOTAL - N_ANT  # 40 nodes survive the antarctica mask
H, W = 4, 5  # toy grid for compute_climatology


# --------------------------------------------------------------------------
# fixed seeded inputs -- one source of truth for capture and for verification
# --------------------------------------------------------------------------


class _FakeDataPred:
    def __init__(self, lon, lat, basins):
        self.land = [np.stack([lon, lat], axis=1)]
        self.land_to_basin_ids = basins


class _FakeDatamodule:
    """Only the attributes basin_wide_mean_twsa actually touches."""

    def __init__(self, lon, lat, basins, antarctica):
        self.data_pred = _FakeDataPred(lon, lat, basins)
        self._antarctica = antarctica

    def get_antarctica_landmask(self):
        return self._antarctica


class _FakeTrain:
    def __init__(self, lsm):
        self.land_ocean_mask = lsm


class _FakeSeqData:
    """Only what get_climatology may touch: the datamodule's C and the train land mask."""

    def __init__(self, climatology, lsm):
        self.twsa_climatology = climatology
        self.land_ocean_mask = lsm
        self.data_train = _FakeTrain(lsm)


def make_inputs():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    d = {}

    # ---- gridded predictions [B, N, T] ------------------------------------
    d["y_hat"] = torch.randn(B, N, T)
    d["y_true"] = torch.randn(B, N, T)

    # same tensors with NaNs punched into *both* operands at different cells:
    # this is the case #30 is about.
    y_hat_nan = d["y_hat"].clone()
    y_true_nan = d["y_true"].clone()
    y_hat_nan[:, :7, :] = float("nan")  # model A is missing the first 7 nodes
    y_true_nan[:, 5:9, :] = float("nan")  # the target is missing nodes 5..8
    d["y_hat_nan"], d["y_true_nan"] = y_hat_nan, y_true_nan

    # a second model, missing a *different* set of cells (the Li et al. fcast)
    fcast_nan = torch.randn(B, N, T)
    fcast_nan[:, 30:, :] = float("nan")
    d["fcast_nan"] = fcast_nan

    # ---- latitude weights [N], normalised to sum to 1 ---------------------
    lat = torch.linspace(-55.0, 70.0, N)
    w = torch.cos(torch.deg2rad(lat))
    d["lat_weights"] = w / (w.sum() + 1e-12)

    # _corr no longer has a case, but these two draws must stay: the stored
    # baselines were captured with them in the stream, so dropping them would
    # reshuffle every torch.randn below and break the pins.
    d["corr_pred"] = torch.randn(B, 6)
    d["corr_true"] = torch.randn(B, 6)

    # ---- nse: [T, N] ------------------------------------------------------
    d["nse_pred"] = torch.randn(T, N)
    d["nse_true"] = torch.randn(T, N)

    # ---- recover_month_index: [B, T] --------------------------------------
    months = torch.arange(B * T).reshape(B, T) % 12
    angle = months.float() * (2 * np.pi / 12)
    d["month_sin"], d["month_cos"] = torch.sin(angle), torch.cos(angle)

    # ---- compute_climatology ----------------------------------------------
    times = pd.date_range("2003-01-01", periods=36, freq="MS")
    d["clim_da"] = xr.DataArray(
        rng.standard_normal((36, H, W)).astype("float32"),
        dims=("time", "lat", "lon"),
        coords={"time": times},
    )
    # land/sea mask, identical for every time step so the reshape(12, -1) holds
    one = rng.random((H, W)) > 0.35
    d["clim_lsm"] = np.broadcast_to(one, (36, H, W)).copy()
    d["n_land"] = int(one.sum())

    # ---- get_months_output_sequences: [B, N, 12] --------------------------
    d["months_input"] = torch.randint(0, 12, (B, N, 12))

    # ---- basin geometry ---------------------------------------------------
    antarctica = np.zeros(N_TOTAL, dtype=bool)
    antarctica[N:] = True  # the last 10 nodes are antarctica
    d["antarctica"] = antarctica
    d["lon"] = rng.uniform(-180, 180, N_TOTAL)
    d["lat_nodes"] = rng.uniform(-90, 90, N_TOTAL)
    # 8 basins over the 40 surviving nodes, deliberately unsorted ids
    d["basins"] = rng.choice(
        np.array([701, 105, 930, 220, 660, 411, 888, 313]), size=N_TOTAL
    )
    d["datamodule"] = _FakeDatamodule(
        d["lon"], d["lat_nodes"], d["basins"], antarctica
    )

    # basin_wide_mean_twsa operands: [T, N]
    d["bw_pred"] = torch.randn(T, N)
    d["bw_true"] = torch.randn(T, N)
    d["bw_clim"] = rng.standard_normal((T, N)).astype("float32")

    bw_pred_nan = d["bw_pred"].clone()
    bw_pred_nan[:, :7] = float("nan")
    d["bw_pred_nan"] = bw_pred_nan

    # get_basin_wide_mean_twsa operands
    d["climatology"] = rng.standard_normal((12, N)).astype("float32")
    d["months_output"] = (
        torch.arange(B * N * T).reshape(B, N, T) % 12
    )
    return d


# --------------------------------------------------------------------------
# the reference calls -- identical code path for capture and verification
# --------------------------------------------------------------------------

HIGNN, COMPARE, CONVLSTM = "evaluation_HiGNN", "compare_models", "evaluation_ConvLSTM"


def _bw(m, d, pred):
    p, t, c, idx = m.basin_wide_mean_twsa(pred, d["bw_true"], d["bw_clim"], d["datamodule"])
    return {"pred": p, "true": t, "clim": c, "ids": np.asarray(idx)}


def _gbw(m, d):
    p, t, c, mo, ids = m.get_basin_wide_mean_twsa(
        d["y_hat"], d["y_true"], d["climatology"], d["months_output"], d["datamodule"]
    )
    return {"pred": p, "true": t, "clim": c, "months": mo, "ids": np.asarray(ids)}


def _time_gbw(m, d, repeats=3):
    """Best-of-`repeats` wall clock for get_basin_wide_mean_twsa (#18)."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        _gbw(m, d)
        best = min(best, time.perf_counter() - t0)
    return best


# name -> (call, scripts that define it)
CASES = {
    "_rmse.clean": (lambda m, d: m._rmse(d["y_hat"], d["y_true"]), (HIGNN, COMPARE, CONVLSTM)),
    "_rmse.nan": (lambda m, d: m._rmse(d["y_hat_nan"], d["y_true_nan"]), (HIGNN, COMPARE, CONVLSTM)),
    "_lat_weighted_rmse.clean": (
        lambda m, d: m._lat_weighted_rmse(d["y_hat"], d["y_true"], d["lat_weights"]),
        (COMPARE,),
    ),
    # _corr is deliberately no longer bit-exact with the pre-move version: it takes
    # its lead count from the input and rejects the transposed [leads, windows]
    # orientation (#12). tests/test_corr_orientation.py owns its behaviour now.
    "nse": (lambda m, d: m.nse(d["nse_pred"], d["nse_true"]), (HIGNN, CONVLSTM)),
    "recover_month_index": (
        lambda m, d: m.recover_month_index(d["month_sin"], d["month_cos"]),
        (HIGNN, COMPARE, CONVLSTM),
    ),
    "compute_climatology.masked": (
        lambda m, d: m.compute_climatology(d["clim_da"], d["clim_lsm"]),
        (HIGNN, COMPARE),
    ),
    # the ConvLSTM copy took an `lsm` argument and never read it -- the unified
    # helper spells that out as lsm=None (gridded climatology, [12, H, W])
    "compute_climatology.gridded": (
        lambda m, d: m.compute_climatology(d["clim_da"], None),
        (CONVLSTM,),
    ),
    "get_months_output_sequences": (
        lambda m, d: m.get_months_output_sequences(d["months_input"], T),
        (HIGNN, COMPARE),
    ),
    "basin_wide_mean_twsa.clean": (lambda m, d: _bw(m, d, d["bw_pred"]), (HIGNN, COMPARE)),
    "basin_wide_mean_twsa.nan": (lambda m, d: _bw(m, d, d["bw_pred_nan"]), (HIGNN, COMPARE)),
    "get_basin_wide_mean_twsa": (_gbw, (HIGNN, COMPARE)),
}

# helpers whose result must be unchanged by the move
BITEXACT = [
    "nse",
    "recover_month_index",
    "compute_climatology.masked",
    "compute_climatology.gridded",
    "get_months_output_sequences",
]


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return torch.as_tensor(np.asarray(x))


def _flatten(name, value):
    if isinstance(value, dict):
        return {f"{name}.{k}": _to_tensor(v) for k, v in value.items()}
    return {name: _to_tensor(value)}


# --------------------------------------------------------------------------
# capture (pre-move)
# --------------------------------------------------------------------------


def capture():
    import importlib

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    d = make_inputs()
    mods = {n: importlib.import_module(n) for n in (HIGNN, COMPARE, CONVLSTM)}

    for script, mod in mods.items():
        out = {}
        for name, (call, scripts) in CASES.items():
            if script not in scripts:
                continue
            out.update(_flatten(name, call(mod, d)))
        if script in CASES["get_basin_wide_mean_twsa"][1]:
            out["timing.get_basin_wide_mean_twsa"] = torch.tensor(_time_gbw(mod, d))
            print(f"  {script} get_basin_wide_mean_twsa: "
                  f"{out['timing.get_basin_wide_mean_twsa'].item() * 1e3:.1f} ms")
        path = BASELINE_DIR / f"{script}.pt"
        torch.save(out, path)
        print(f"captured {len(out):2d} tensors from {script} -> {path}")


# --------------------------------------------------------------------------
# verification (post-move)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data():
    return make_inputs()


@pytest.fixture(scope="module")
def baselines():
    out = {}
    for script in (HIGNN, COMPARE, CONVLSTM):
        path = BASELINE_DIR / f"{script}.pt"
        if not path.exists():
            pytest.skip(f"no pre-move baseline at {path}; run `python tests/test_m0.py capture` first")
        out[script] = torch.load(path, weights_only=False)
    return out


@pytest.fixture(scope="module")
def em():
    from utils import eval_metrics

    return eval_metrics


def _new(name, em, d):
    call, _ = CASES[name]
    return _flatten(name, call(em, d))


# ---- the move itself: bit-exact -------------------------------------------


@pytest.mark.parametrize("name", BITEXACT)
def test_move_is_bit_exact(name, em, data, baselines):
    _, scripts = CASES[name]
    new = _new(name, em, data)
    for script in scripts:
        for key, value in new.items():
            old = baselines[script][key]
            assert old.dtype == value.dtype, f"{key}: dtype {old.dtype} -> {value.dtype} ({script})"
            assert torch.equal(old, value), f"{key} changed vs {script} -- not one of the four changes"


# get_climatology is deliberately no longer bit-exact with the pre-move version:
# that one unnormalized the train slice, which is ~0 in TWSA anomaly mode, and
# cached it in climatology.pt. It now returns the datamodule's raw training C.


def test_get_climatology_returns_the_datamodules_c(em, data, tmp_path):
    """C [12, H, W], float64 like TWSA on disk, land-flattened to float32 [12, N_land]."""
    clim = data["clim_da"].groupby("time.month").mean("time").astype("float64")
    lsm = xr.DataArray(data["clim_lsm"], dims=("time", "lat", "lon"))
    seq = _FakeSeqData(clim, lsm)
    config = {"experiment": {"save_dir": str(tmp_path)}}
    expected = clim.values[:, data["clim_lsm"][0]].astype("float32")

    new = np.asarray(em.get_climatology(seq, config))
    assert new.dtype == np.float32
    assert np.array_equal(new, expected)
    assert not (tmp_path / "climatology.pt").exists(), "no cache is written"

    torch.save(np.zeros_like(expected), tmp_path / "climatology.pt")  # a stale cache
    assert np.array_equal(np.asarray(em.get_climatology(seq, config)), expected)


def test_the_three_scripts_no_longer_define_the_helpers():
    """The move is a move, not a copy."""
    import importlib

    moved = set(CASES) | {"get_climatology"}
    names = {n.split(".")[0] for n in moved}
    for script in (HIGNN, COMPARE, CONVLSTM):
        mod = importlib.import_module(script)
        for fn in names:
            if not hasattr(mod, fn):
                continue
            defined_here = getattr(mod, fn).__module__ == script
            assert not defined_here, f"{script} still defines {fn}"


def test_mape_and_smape_are_gone():
    import importlib

    for script in (HIGNN, COMPARE, CONVLSTM):
        mod = importlib.import_module(script)
        assert not hasattr(mod, "_mape"), f"{script} still has _mape"
        assert not hasattr(mod, "_smape"), f"{script} still has _smape"
    from utils import eval_metrics

    assert not hasattr(eval_metrics, "_mape")
    assert not hasattr(eval_metrics, "_smape")


# ---- #30: one reduction policy, one denominator ---------------------------


def test_30_rmse_clean_input_matches_both_old_copies(em, data, baselines):
    """With no NaNs the mean/nanmean split was invisible -- it must stay invisible."""
    new = em._rmse(data["y_hat"], data["y_true"])
    for script in (HIGNN, COMPARE, CONVLSTM):
        old = baselines[script]["_rmse.clean"]
        print(f"#30 clean  {script:20s} old={old.mean():.10f}  new={new.mean():.10f}")
        assert torch.equal(old, new)


def test_30_rmse_is_nan_aware(em, data, baselines):
    old_mean = baselines[HIGNN]["_rmse.nan"]  # torch.mean copy
    old_nanmean = baselines[COMPARE]["_rmse.nan"]  # torch.nanmean copy
    new = em._rmse(data["y_hat_nan"], data["y_true_nan"])

    print(f"#30 nan  torch.mean copy  -> {old_mean.mean().item()}")
    print(f"#30 nan  torch.nanmean    -> {old_nanmean.mean().item():.10f}")
    print(f"#30 nan  new              -> {new.mean().item():.10f}")

    assert torch.isnan(old_mean).all(), "precondition: the torch.mean copy poisoned every score"
    assert torch.isfinite(new).all(), "the unified reduction must survive NaNs"
    assert torch.allclose(new, old_nanmean, equal_nan=True)


def test_30_both_models_share_one_denominator(em, data):
    """Two models with different NaN footprints must be scored on the intersection."""
    a, b, y = data["y_hat_nan"], data["fcast_nan"], data["y_true_nan"]

    # per-model nan-awareness alone gives the two models *different* denominators
    solo_a = em.valid_mask(a, y).sum(dim=1)
    solo_b = em.valid_mask(b, y).sum(dim=1)
    assert not torch.equal(solo_a, solo_b), "precondition: the footprints differ"

    mask = em.valid_mask(a, b, y)
    shared = mask.sum(dim=1)
    rmse_a = em._rmse(a, y, mask=mask)
    rmse_b = em._rmse(b, y, mask=mask)
    assert torch.isfinite(rmse_a).all() and torch.isfinite(rmse_b).all()

    # both scores are sum-of-squares over the shared cells / the shared count
    for model, score in ((a, rmse_a), (b, rmse_b)):
        ref = torch.sqrt(((y - model) ** 2).masked_fill(~mask, 0.0).sum(dim=1) / shared)
        assert torch.equal(score, ref)
    print(
        f"#30 denominator (lead 1): model A alone={solo_a[0, 0].item()}  "
        f"model B alone={solo_b[0, 0].item()}  shared={shared[0, 0].item()}"
    )

    cov = em.mask_coverage(mask)
    print(f"#30 coverage per lead: {cov.mean(dim=0).tolist()}")
    assert cov.shape == (data["y_hat"].shape[0], T)
    assert (cov <= 1.0).all() and (cov > 0).all()

    # scoring model A on the shared mask differs from scoring it on its own mask
    assert not torch.allclose(rmse_a, em._rmse(a, y))


# ---- #14: one NaN policy for gridded metrics and basin means --------------


def test_14_basin_means_and_gridded_metrics_use_the_same_cells(em, data):
    """nan_to_num is gone: a NaN cell is excluded from the basin mean, not zeroed."""
    pred_nan = data["bw_pred_nan"]
    zero_filled = torch.nan_to_num(pred_nan)

    p_excl, _, _, ids = em.basin_wide_mean_twsa(
        pred_nan, data["bw_true"], data["bw_clim"], data["datamodule"]
    )
    p_zero, _, _, _ = em.basin_wide_mean_twsa(
        zero_filled, data["bw_true"], data["bw_clim"], data["datamodule"]
    )
    print(f"#14 basin means, first basin: excluded={p_excl[0, 0]:.6f}  nan_to_num={p_zero[0, 0]:.6f}")
    assert not np.allclose(p_excl, p_zero), "the old zero-fill and the new policy must differ"

    # and the exclusion policy is the same one _rmse uses: mean over finite cells
    basins = np.asarray(data["basins"])[~data["antarctica"]]
    first = np.asarray(ids)[0]
    sel = basins == first
    row = pred_nan[0].numpy()[sel]
    assert np.isclose(p_excl[0, 0], np.nanmean(row), rtol=0, atol=1e-6)
    assert np.isnan(row).any(), "precondition: this basin does contain a NaN cell"


# ---- #6: sqrt(sum(w*e^2)), not sqrt(mean(w*e^2)) --------------------------


def test_6_lat_weighted_rmse_drops_the_second_division(em, data, baselines):
    old = baselines[COMPARE]["_lat_weighted_rmse.clean"]
    new = em._lat_weighted_rmse(data["y_hat"], data["y_true"], data["lat_weights"])
    ratio = (new / old)
    print(f"#6 old={old.mean():.10f}  new={new.mean():.10f}  ratio={ratio.mean():.10f}  sqrt(N)={np.sqrt(N):.10f}")
    assert torch.allclose(ratio, torch.full_like(ratio, float(np.sqrt(N))), rtol=1e-5)

    # and it is literally sqrt(sum(w*e^2)) over the node axis
    ref = torch.sqrt((((data["y_true"] - data["y_hat"]) ** 2) * data["lat_weights"][None, :, None]).sum(dim=1))
    assert torch.allclose(new, ref, rtol=0, atol=1e-7)


# ---- #18: one grouped reduction instead of B*6 DataFrames -----------------


def test_18_basin_means_unchanged_and_faster(em, data, baselines):
    new = _new("basin_wide_mean_twsa.clean", em, data)
    for key, value in new.items():
        old = baselines[HIGNN][key]
        if key.endswith(".ids"):
            assert torch.equal(old, value), "basin order changed"
            continue
        diff = (old.double() - value.double()).abs().max().item()
        print(f"#18 {key}: max|old-new| = {diff:.3e}")
        assert diff < 1e-6, f"{key} moved by {diff}"

    old_s = baselines[HIGNN]["timing.get_basin_wide_mean_twsa"].item()
    new_s = _time_gbw(em, data)
    print(
        f"#18 get_basin_wide_mean_twsa ({B} windows x {T} leads, {N} nodes): "
        f"old={old_s * 1e3:.1f} ms  new={new_s * 1e3:.1f} ms  speedup={old_s / new_s:.1f}x"
    )
    assert new_s < old_s


def test_18_get_basin_wide_mean_twsa_matches(em, data, baselines):
    new = _new("get_basin_wide_mean_twsa", em, data)
    for key, value in new.items():
        old = baselines[HIGNN][key]
        if key.endswith(".ids") or key.endswith(".months"):
            assert torch.equal(old, value)
            continue
        diff = (old.double() - value.double()).abs().max().item()
        print(f"#18 {key}: max|old-new| = {diff:.3e}")
        assert diff < 1e-6


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "capture":
        capture()
    else:
        raise SystemExit("usage: python tests/test_m0.py capture")
