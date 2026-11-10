"""Helpers of the per-basin anomaly-skill evaluation (scripts/basin_anomaly_skill.py)."""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import basin_anomaly_skill as bas


def _toy(seed=0, B=40, H=6, K=3):
    rng = np.random.default_rng(seed)
    clim = rng.normal(size=(B, H, K)) * 5
    truth = clim + rng.normal(size=(B, H, K))
    return clim, truth


def test_last_input_anomaly_uses_the_month_before_each_start():
    times = pd.date_range("2000-01-01", "2001-12-01", freq="MS").values
    anom = np.arange(len(times), dtype=float)[:, None] * np.array([1.0, 10.0])  # [T, 2]
    starts = pd.DatetimeIndex(["2000-03-01", "2001-01-01"]).values

    a0 = bas.last_input_anomaly(anom, times, starts)

    # 2000-02 is index 1, 2000-12 is index 11
    np.testing.assert_allclose(a0, [[1.0, 10.0], [11.0, 110.0]])


def test_perfect_forecast_has_acc_one_and_skill_one():
    clim, truth = _toy()
    np.testing.assert_allclose(bas.anomaly_correlation(truth, truth, clim), 1.0)
    rmse_clim = bas.rmse(clim, truth)
    np.testing.assert_allclose(bas.skill_score(bas.rmse(truth, truth), rmse_clim), 1.0)


def test_acc_ignores_amplitude_and_offset_but_skill_does_not():
    clim, truth = _toy()
    pred = clim + 3 * (truth - clim) + 2

    np.testing.assert_allclose(bas.anomaly_correlation(pred, truth, clim), 1.0)
    assert (bas.skill_score(bas.rmse(pred, truth), bas.rmse(clim, truth)) < 0).all()


def test_climatology_forecast_has_zero_skill_against_climatology():
    clim, truth = _toy()
    np.testing.assert_allclose(bas.skill_score(bas.rmse(clim, truth), bas.rmse(clim, truth)), 0.0)


def test_persistence_is_perfect_for_a_constant_anomaly():
    clim, _ = _toy()
    offset = np.array([1.5, -2.0, 0.3])  # one constant anomaly per basin
    truth = clim + offset
    a0 = np.broadcast_to(offset, (clim.shape[0], clim.shape[2]))

    pers = bas.persistence_forecast(clim, a0)

    np.testing.assert_allclose(pers, truth)


def test_moving_block_indices_are_contiguous_runs_of_the_block_length():
    n, block = 32, 12
    idx = bas.moving_block_indices(n, block, n_boot=50, rng=np.random.default_rng(1))

    assert idx.shape == (50, n)
    assert idx.min() >= 0 and idx.max() < n
    for row in idx:
        for start in range(0, n, block):
            run = row[start:start + block]
            np.testing.assert_array_equal(np.diff(run), 1)


def _skill_table():
    """Long table as in main: value encodes basin, metric and lead; "" marks the basins outside the test set."""
    rows = [dict(basin=basin, lead=lead, metric=metric, value=b + m / 10 + lead / 100)
            for b, basin in enumerate(["Nile", "Amazon", "", ""])
            for m, metric in enumerate(["skill_clim", "acc", "skill_pers", "acc_persistence"])
            for lead in range(1, 7)]
    return pd.DataFrame(rows)


def test_latex_table_has_three_metric_rows_per_basin_with_leads_as_columns():
    tex = bas.latex_table(_skill_table(), ["Amazon", "Nile"], n_starts=32).splitlines()

    assert "Basin & Metric & L1 & L2 & L3 & L4 & L5 & L6 \\\\" in tex
    assert "\\begin{tabular}{llcccccc}" in tex
    assert "\\renewcommand{\\baselinestretch}{1}\\scriptsize" in tex  # 49 rows on one manuscript page
    body =[line for line in tex if line.endswith("\\\\")][1:]
    assert len(body) == 6
    # given basin order, then r, skill vs. persistence, skill vs. climatology
    assert body[0].startswith("\\multirow{3}{*}{Amazon} & $r_\\mathrm{anom}$ & $1.11$ & $1.12$")
    assert body[0].endswith("$1.16$ \\\\")
    assert body[1].startswith(" & SS$_\\mathrm{pers}$ & $1.21$")
    assert body[2].startswith(" & SS$_\\mathrm{clim}$ & $1.01$")
    assert body[3].startswith("\\multirow{3}{*}{Nile} & $r_\\mathrm{anom}$ & $0.11$")
    assert tex.count("\\middlehline") == 2


def test_grace_results_go_to_a_separate_directory():
    assert bas.out_dir("test") == bas.OUT
    assert bas.out_dir("grace") == bas.OUT + "_grace"


def test_latex_table_names_the_dataset_and_the_number_of_starts():
    test = bas.latex_table(_skill_table(), ["Nile"], n_starts=32)
    grace = bas.latex_table(_skill_table(), ["Nile"], n_starts=58, dataset="grace")

    assert "on the test set (mean over 8 seeds, 32 forecast starts)" in test
    assert "\\label{tab:basin_anomaly_skill}" in test
    assert "on GRACE/-FO (mean over 8 seeds, 58 forecast starts)" in grace
    assert "\\label{tab:basin_anomaly_skill_grace}" in grace


def test_grace_summary_is_written_to_the_grace_directory(tmp_path, monkeypatch):
    rows = [dict(basin="Nile", lead=lead, metric=metric, value=0.1, ci_lo=0.0, ci_hi=0.2)
            for lead in range(1, 7)
            for metric in ("skill_pers", "skill_clim", "acc", "acc_persistence", "skill_clim_persistence")]
    monkeypatch.setattr(bas, "OUT", str(tmp_path / "skill"))
    (tmp_path / "skill_grace").mkdir()

    bas.write_summary(pd.DataFrame(rows), 58, pd.date_range("2009-10-01", periods=58, freq="MS").values,
                      dataset="grace")

    header = (tmp_path / "skill_grace" / "summary.md").read_text().splitlines()[0]
    assert header.endswith("GRACE/-FO")
    assert not (tmp_path / "skill").exists()


def test_summary_table_with_leads_as_columns(tmp_path, monkeypatch):
    """Counts are 'skill > 0 (ci_lo > 0)' over all basins, medians over all basins, model only."""
    basins = ["Nile", "Amazon", "", ""]
    pers_v, pers_lo = [0.3, 0.2, -0.1, 0.4], [0.1, -0.1, -0.3, 0.2]
    clim_base, acc_v = [0.6, -0.2, -0.1, 0.3], [0.9, 0.7, 0.5, 0.3]
    rows = []
    for b, basin in enumerate(basins):
        for lead in range(1, 7):
            clim_v = clim_base[b] - 0.1 * (lead - 1)  # skill vs. climatology falls with lead
            for metric, v, lo in (("skill_pers", pers_v[b], pers_lo[b]), ("skill_clim", clim_v, clim_v - 0.35),
                                  ("acc", acc_v[b], acc_v[b] - 0.2), ("acc_persistence", 0.0, -0.1),
                                  ("skill_clim_persistence", 0.0, -0.1)):
                rows.append(dict(basin=basin, lead=lead, metric=metric, value=v, ci_lo=lo, ci_hi=v + 0.1))
    monkeypatch.setattr(bas, "OUT", str(tmp_path))

    bas.write_summary(pd.DataFrame(rows), 32, pd.date_range("2017-06-01", periods=32, freq="MS").values)

    lines = (tmp_path / "summary.md").read_text().splitlines()
    assert "| | L1 | L2 | L3 | L4 | L5 | L6 |" in lines

    def cells(label):
        row = [line for line in lines if line.startswith(f"| {label} |")]
        assert len(row) == 1, label
        return [c.strip() for c in row[0].strip("|").split("|")][1:]

    assert cells("Basins better than persistence (interval > 0)") == ["3 (2)"] * 6
    clim = cells("Basins better than climatology (interval > 0)")
    assert (clim[0], clim[5]) == ("2 (1)", "1 (0)")
    assert cells("Median anomaly r, model") == ["0.60"] * 6
    sc = cells("Median skill vs climatology, model")
    assert (sc[0], sc[5]) == ("0.10", "-0.40")
    assert cells("Median skill vs persistence, model") == ["0.25"] * 6


def test_start_dates_follow_the_dataset_sample_rule():
    times = pd.date_range("2016-06-01", "2020-06-01", freq="MS").values
    starts = bas.start_dates(times, seq_len=12, pred_len=6)
    assert pd.Timestamp(starts[0]) == pd.Timestamp("2017-06-01")
    assert pd.Timestamp(starts[-1]) == pd.Timestamp("2020-01-01")  # last lead on 2020-06
    assert len(starts) == 32


def test_start_dates_defaults_are_the_model_window():
    times = pd.date_range("2016-06-01", "2020-06-01", freq="MS").values
    assert (bas.SEQ_LEN, bas.PRED_LEN) == (12, 6)
    np.testing.assert_array_equal(bas.start_dates(times), bas.start_dates(times, seq_len=12, pred_len=6))


def test_start_dates_reject_a_gap():
    times = pd.date_range("2016-06-01", "2020-06-01", freq="MS").delete(10).values
    with pytest.raises(AssertionError):
        bas.start_dates(times)
