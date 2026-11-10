"""compare_models.plot_basintwsa_timesteps reports the anomaly correlation (shared climatology) and RMSE."""
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import compare_models


def _r_anom(pred, true, clim):
    """Pearson r of pred - clim and true - clim over the windows, per lead, then the lead mean."""
    pa, ta = pred - clim, true - clim
    pa, ta = pa - pa.mean(0), ta - ta.mean(0)
    return ((pa * ta).sum(0) / np.sqrt((pa ** 2).sum(0) * (ta ** 2).sum(0))).mean()


def test_box_shows_r_anom_against_the_shared_climatology_and_rmse():
    rng = np.random.default_rng(0)
    B = 40
    clim = 5 * np.sin(np.arange(B)[:, None] + np.arange(6)[None, :])  # [B, 6], a seasonal-like signal
    true = clim + rng.normal(size=(B, 6))
    pred = clim + rng.normal(size=(B, 6))          # seasonal cycle right, anomalies unrelated
    fcast = clim + (true - clim) + 0.5              # anomalies right, constant offset
    # basin-sliced arrays of evaluate() carry a trailing basin axis of size 1
    pred, true, fcast, clim = (x[:, :, None].astype(np.float32) for x in (pred, true, fcast, clim))

    fig, ax = plt.subplots()
    compare_models.plot_basintwsa_timesteps("X", 1, pred, true, fcast, ax=ax, clim=clim)
    text = ax.texts[0].get_text()
    plt.close(fig)

    exp_pred = _r_anom(pred[..., 0], true[..., 0], clim[..., 0])
    rmse_pred = np.sqrt(((pred - true) ** 2).mean(0)).mean()
    assert text == (f"$r_{{anom}}$={exp_pred:.2f}/1.00\n"
                    f"RMSE={rmse_pred:.2f}/0.50 cm")
    assert abs(exp_pred) < 0.5  # full-signal r would be ~0.97 here, so the box must not show it
