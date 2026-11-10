"""plot_latent_removal.py: per-lead effect of replacing an LSTM input by a constant, from scores.csv."""

import pathlib
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "plotting"))

import plot_latent_removal as plr
from run_latent_removal import COLUMNS


def _scores(seeds=(("45", 2.0), ("46", 3.0))):
    """Two seeds and two leads. RMSE of seed 46 is 1 cm higher, so unpaired deltas would be wrong."""
    rows = []
    for seed, base in seeds:
        for dataset, scale in (("test", 1), ("val", 10)):
            for lead in range(2):
                rows.append(("full", seed, dataset, "none", "none", lead, base))
                rows.append(("full", seed, dataset, "lstm_input", "twsa", lead, base + scale * 0.8 * (lead + 1)))
                rows.append(("full", seed, dataset, "lstm_input", "lead", lead, base + scale * 0.1 * (lead + 1)))
                for i, removal in enumerate(("dim1", "dim2", "dim3", "dim4", "all")):
                    delta = scale * 0.01 * (i + 1) * (lead + 1)
                    rows.append(("full", seed, dataset, "channel_mean", removal, lead, base + delta))
                    rows.append(("full", seed, dataset, "cell_mean", removal, lead, base + 5))
    rows += [("skip_gnn", "45", "test", "none", "none", lead, 2.0) for lead in range(2)]
    rows += [("skip_gnn", "45", "test", "channel_mean", "dim1", lead, 9.0) for lead in range(2)]
    return pd.DataFrame(rows, columns=COLUMNS)


def test_inputs_are_the_six_flattened_inputs():
    assert list(plr.INPUTS.values()) == ["TWSA history", "Lead embedding", "Dim 1", "Dim 2", "Dim 3", "Dim 4"]


def test_deltas_are_per_lead_on_test_and_paired_to_the_same_seed():
    deltas = plr.flat_deltas(_scores())

    assert deltas.loc[("TWSA history", "45")].to_numpy() == pytest.approx([0.8, 1.6])
    assert deltas.loc[("TWSA history", "46")].to_numpy() == pytest.approx([0.8, 1.6])
    assert deltas.loc[("Lead embedding", "46")].to_numpy() == pytest.approx([0.1, 0.2])
    assert deltas.loc[("Dim 3", "45")].to_numpy() == pytest.approx([0.03, 0.06])


def test_deltas_hold_every_seed_of_the_six_inputs_and_nothing_else():
    deltas = plr.flat_deltas(_scores())

    assert set(deltas.index) == {(label, seed) for label in plr.INPUTS.values() for seed in ("45", "46")}


def test_figure_is_written(tmp_path):
    path = tmp_path / "figure.png"

    plr.plot(plr.flat_deltas(_scores()), str(path))

    assert path.stat().st_size > 0


def test_other_seeds_of_the_latents_get_one_pastel1_color_each(tmp_path):
    plr.plot(plr.flat_deltas(_scores((("45", 2.0), ("46", 3.0), ("47", 4.0)))), str(tmp_path / "figure.png"))

    pastel = [mcolors.to_hex(c) for c in plt.get_cmap("Pastel1").colors]
    for ax in plt.gcf().axes:
        thin = [mcolors.to_hex(line.get_color()) for line in ax.get_lines() if line.get_linewidth() < 1]
        seed_colors = [c for c in thin if c in pastel]
        # two other seeds, each drawn once for every one of the four dims
        assert len(set(seed_colors)) == 2
        assert all(seed_colors.count(c) == 4 for c in set(seed_colors))
