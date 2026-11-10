"""Per-basin ablation table (scripts/basin_ablation.py)."""
import pathlib
import sys

import numpy as np
import pytest
from scipy.stats import ttest_ind

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import basin_ablation as ba


def toy_rmse():
    """rmse [models 2, seeds 3, leads 2, basins 3]; basin 3 has a NaN and must be dropped."""
    rng = np.random.default_rng(0)
    rm = 5 + rng.random((2, 3, 2, 3))
    rm[1, :, :, 0] += 1.0  # ablation clearly worse in basin 0
    rm[0, 1, 0, 2] = np.nan
    return rm


def test_rows_give_seed_mean_difference_per_lead_and_lead_mean():
    rm = toy_rmse()
    df = ba.basin_rows(rm, ["Full", "No X"], np.array([2030000010, 6030000020, 7030000030]),
                       {2030000010: "Danube"})

    assert list(df["basin"]) == [2030000010, 6030000020]  # NaN basin dropped
    assert list(df["model"]) == ["No X", "No X"]
    row = df.iloc[0]
    assert row["name"] == "Danube" and row["continent"] == "Europe"
    assert df.iloc[1]["name"] == "" and df.iloc[1]["continent"] == "S America"
    for lead in range(2):
        assert row[f"L{lead + 1}"] == pytest.approx(rm[1, :, lead, 0].mean() - rm[0, :, lead, 0].mean())
    abl, full = rm[1, :, :, 0].mean(1), rm[0, :, :, 0].mean(1)
    assert row["Mean"] == pytest.approx(abl.mean() - full.mean())
    assert row["p"] == pytest.approx(ttest_ind(abl, full, equal_var=False).pvalue)


def test_significant_keeps_lead_mean_p_below_005():
    rm = toy_rmse()
    df = ba.basin_rows(rm, ["Full", "No X"], np.array([2030000010, 6030000020, 7030000030]), {})
    sig = ba.significant(df)
    assert (sig["p"] < 0.05).all()
    assert 2030000010 in set(sig["basin"])
    assert set(sig["basin"]) == set(df.loc[df["p"] < 0.05, "basin"])


def test_markdown_and_latex_have_one_line_per_basin_and_model():
    rm = np.concatenate([toy_rmse(), toy_rmse()[1:]])  # 2 ablations
    df = ba.basin_rows(rm, ["Full", "No X", "No Y"], np.array([2030000010, 6030000020, 7030000030]), {})
    md, tex = ba.markdown(df), ba.latex(df)
    for text in (md, tex):
        assert "No X" in text and "No Y" in text
        assert text.count("2030000010") >= 2 and text.count("6030000020") >= 2
    assert "\\begin{longtable}" in tex
