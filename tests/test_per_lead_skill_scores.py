"""Area-weighted NSE, KGE'' and r per lead -- the columns `per_lead_skill_scores`
adds to per_timestep_scores.csv.

Every expected value is computed here with plain loops over (node, lead), one
series over the windows at a time, so the helper is checked against an
independent implementation rather than against its own internals:

- NSE is pooled: 1 - sum w*SSE / sum w*SSE_ref over all cells and windows.
- r and KGE'' are per pixel, then averaged with the weights of the pixels whose
  score is finite.
- KGE'' (Tang et al., 2021) = 1 - sqrt((r-1)^2 + (alpha-1)^2 + beta''^2), with
  alpha = sd_pred/sd_obs, beta'' = (mean_pred - mean_obs)/sd_obs, population sd.
"""

import warnings

import numpy as np
import pytest
import torch
from scores.continuous import kge

from utils.eval_metrics import gridded_score, per_lead_skill_scores

COLUMNS = {'nse_latw', 'nse_deseas_latw', 'kge2_latw', 'kge2_deseas_latw',
           'corr_latw', 'corr_deseas_latw', 'clim_nse_latw', 'clim_kge2_latw', 'clim_corr_latw'}


def _fields(num_windows=40, num_nodes=5, num_leads=3, seed=0):
    """pred, true, climatology [windows, nodes, leads] and made-up node weights [nodes]."""
    rng = np.random.default_rng(seed)
    target_month = np.arange(num_windows)[:, None, None] + np.arange(num_leads)[None, None, :]
    amplitude = rng.uniform(2, 10, (1, num_nodes, 1))
    clim = amplitude * np.sin(2 * np.pi * target_month / 12) + rng.normal(0, 3, (1, num_nodes, 1))
    clim = np.broadcast_to(clim, (num_windows, num_nodes, num_leads)).copy()
    true = clim + rng.normal(0.5, rng.uniform(0.5, 3, (1, num_nodes, 1)), clim.shape)
    pred = true + rng.normal(0.2, 1.5, clim.shape)
    weights = rng.uniform(0.2, 1.0, num_nodes)
    return pred, true, clim, weights


def _r(p, t):
    return np.corrcoef(p, t)[0, 1]


def _kge2(p, t):
    beta = (p.mean() - t.mean()) / t.std()
    return 1 - np.sqrt((_r(p, t) - 1) ** 2 + (p.std() / t.std() - 1) ** 2 + beta ** 2)


def _weighted_mean(values, weights):
    """Weighted mean over the pixels with a finite score, the others dropped with their weight."""
    values, weights = np.asarray(values), np.asarray(weights)
    ok = np.isfinite(values)
    return (values[ok] * weights[ok]).sum() / weights[ok].sum()


def _expected(y_hat, y_true, clim, weights):
    """Every column, by loops over (node, lead). No masking: the inputs must be finite."""
    num_nodes, num_leads = y_true.shape[1:]
    out = {name: np.empty(num_leads) for name in COLUMNS}
    with np.errstate(invalid='ignore', divide='ignore'):
        for lead in range(num_leads):
            sse = sst = sse_clim = 0.0
            pixel = {name: [] for name in ('kge2', 'kge2_deseas', 'corr', 'corr_deseas', 'clim_kge2', 'clim_corr')}
            for node in range(num_nodes):
                p, t, c = y_hat[:, node, lead], y_true[:, node, lead], clim[:, node, lead]
                sse += weights[node] * ((p - t) ** 2).sum()
                sst += weights[node] * ((t - t.mean()) ** 2).sum()
                sse_clim += weights[node] * ((c - t) ** 2).sum()
                pixel['kge2'].append(_kge2(p, t))
                pixel['kge2_deseas'].append(_kge2(p - c, t - c))
                pixel['corr'].append(_r(p, t))
                pixel['corr_deseas'].append(_r(p - c, t - c))
                pixel['clim_kge2'].append(_kge2(c, t))
                pixel['clim_corr'].append(_r(c, t))
            out['nse_latw'][lead] = 1 - sse / sst
            out['nse_deseas_latw'][lead] = 1 - sse / sse_clim
            out['clim_nse_latw'][lead] = 1 - sse_clim / sst
            for name, values in pixel.items():
                out[f'{name}_latw'][lead] = _weighted_mean(values, weights)
    return out


def _assert_scores_equal(actual, expected, columns=COLUMNS, rtol=1e-9, atol=1e-12):
    for name in columns:
        np.testing.assert_allclose(np.asarray(actual[name], dtype=np.float64), expected[name],
                                   rtol=rtol, atol=atol, err_msg=name)


# ------------------------------------------------------------------------ the contract


def test_returns_exactly_the_nine_columns_with_one_value_per_lead():
    """The lead axis is not hardwired to 6 -- output_seq_len is a config value."""
    out = per_lead_skill_scores(*_fields(num_leads=3))

    assert set(out) == COLUMNS
    for name, values in out.items():
        assert np.asarray(values).shape == (3,), name


def test_all_columns_match_the_loop_reference():
    pred, true, clim, weights = _fields()
    _assert_scores_equal(per_lead_skill_scores(pred, true, clim, weights),
                         _expected(pred, true, clim, weights))


# ----------------------------------------------------------------------------- NSE


def test_nse_is_pooled_over_pixels_against_each_pixels_mean_over_the_windows():
    pred, true, clim, weights = _fields()
    out = per_lead_skill_scores(pred, true, clim, weights)

    w = weights[None, :, None]
    sse = (w * (pred - true) ** 2).sum(axis=(0, 1))
    sst = (w * (true - true.mean(axis=0, keepdims=True)) ** 2).sum(axis=(0, 1))
    np.testing.assert_allclose(out['nse_latw'], 1 - sse / sst, rtol=1e-9)

    # pooled, not the weighted mean of per-pixel NSEs -- those differ on this data
    per_pixel = 1 - (w * (pred - true) ** 2).sum(0) / (w * (true - true.mean(0)) ** 2).sum(0)
    assert not np.allclose(out['nse_latw'], (weights[:, None] * per_pixel).sum(0) / weights.sum())


def test_nse_deseas_is_the_skill_against_the_climatology():
    pred, true, clim, weights = _fields()
    out = per_lead_skill_scores(pred, true, clim, weights)

    w = weights[None, :, None]
    expected = 1 - (w * (pred - true) ** 2).sum(axis=(0, 1)) / (w * (clim - true) ** 2).sum(axis=(0, 1))
    np.testing.assert_allclose(out['nse_deseas_latw'], expected, rtol=1e-9)


def test_the_climatology_scores_zero_nse_deseas():
    _, true, clim, weights = _fields()
    out = per_lead_skill_scores(clim, true, clim, weights)

    np.testing.assert_allclose(out['nse_deseas_latw'], 0.0, atol=1e-12)


def test_a_perfect_forecast_scores_one():
    _, true, clim, weights = _fields()
    out = per_lead_skill_scores(true, true, clim, weights)

    for name in ('nse_latw', 'nse_deseas_latw', 'corr_latw', 'corr_deseas_latw', 'kge2_latw', 'kge2_deseas_latw'):
        np.testing.assert_allclose(out[name], 1.0, atol=1e-12, err_msg=name)


# ------------------------------------------------------------------------------- r


def test_corr_is_the_weighted_mean_of_the_per_pixel_r():
    pred, true, clim, weights = _fields()
    out = per_lead_skill_scores(pred, true, clim, weights)

    num_nodes, num_leads = true.shape[1:]
    for lead in range(num_leads):
        raw = [np.corrcoef(pred[:, n, lead], true[:, n, lead])[0, 1] for n in range(num_nodes)]
        anom = [np.corrcoef(pred[:, n, lead] - clim[:, n, lead], true[:, n, lead] - clim[:, n, lead])[0, 1]
                for n in range(num_nodes)]
        assert out['corr_latw'][lead] == pytest.approx((weights * raw).sum() / weights.sum(), rel=1e-9)
        assert out['corr_deseas_latw'][lead] == pytest.approx((weights * anom).sum() / weights.sum(), rel=1e-9)


# ---------------------------------------------------------------------------- KGE''


def test_kge2_of_a_constant_offset_is_one_minus_offset_over_sd():
    """y_hat = y_true + c per node gives r = alpha = 1, so KGE'' = 1 - |c| / sd_true."""
    _, true, clim, weights = _fields()
    offset = np.array([0.7, -1.3, 0.0, 2.1, -0.4])
    out = per_lead_skill_scores(true + offset[None, :, None], true, clim, weights)

    per_pixel = 1 - np.abs(offset[:, None]) / true.std(axis=0)                     # [nodes, leads]
    per_pixel_anom = 1 - np.abs(offset[:, None]) / (true - clim).std(axis=0)
    np.testing.assert_allclose(out['kge2_latw'], (weights[:, None] * per_pixel).sum(0) / weights.sum(), rtol=1e-9)
    np.testing.assert_allclose(out['kge2_deseas_latw'],
                               (weights[:, None] * per_pixel_anom).sum(0) / weights.sum(), rtol=1e-9)


def test_kge2_stays_finite_for_a_zero_mean_truth():
    """Standard KGE's beta = mean_pred / mean_obs divides by zero when the pixel's mean is
    0 -- the reason for KGE''. Integer data makes the mean exactly 0.0."""
    rng = np.random.default_rng(3)
    _, _, clim, weights = _fields()
    true = rng.integers(-5, 6, clim.shape).astype(np.float64)
    true[-1] = -true[:-1].sum(axis=0)
    assert (true.mean(axis=0) == 0).all()
    pred = true + rng.normal(0.3, 1.0, true.shape)

    assert not np.isfinite(gridded_score(kge, pred, true).values).all()  # the standard KGE fails here
    out = per_lead_skill_scores(pred, true, clim, weights)
    assert np.isfinite(out['kge2_latw']).all()
    _assert_scores_equal(out, _expected(pred, true, clim, weights), columns=['kge2_latw'])


# ------------------------------------------------------------------------- weighting


def test_scaling_all_weights_changes_nothing():
    pred, true, clim, weights = _fields()
    _assert_scores_equal(per_lead_skill_scores(pred, true, clim, 3 * weights),
                         per_lead_skill_scores(pred, true, clim, weights))


@pytest.mark.parametrize('heavy', [0.1, 1.0, 10.0])
def test_an_error_on_one_node_counts_in_proportion_to_its_weight(heavy):
    """All nodes perfect but node 1: the loss 1 - score is node 1's loss times its weight share."""
    pred, true, clim, _ = _fields()
    pred = true.copy()
    pred[:, 1] += np.random.default_rng(5).normal(0, 2.0, pred[:, 1].shape)
    weights = np.array([1.0, heavy, 1.0, 1.0, 1.0])
    out = per_lead_skill_scores(pred, true, clim, weights)

    share = heavy / weights.sum()
    for lead in range(true.shape[2]):
        p, t = pred[:, 1, lead], true[:, 1, lead]
        assert 1 - out['corr_latw'][lead] == pytest.approx(share * (1 - _r(p, t)), rel=1e-9)
        assert 1 - out['kge2_latw'][lead] == pytest.approx(share * (1 - _kge2(p, t)), rel=1e-9)
        sst = (weights * ((true[:, :, lead] - true[:, :, lead].mean(0)) ** 2).sum(0)).sum()
        assert 1 - out['nse_latw'][lead] == pytest.approx(heavy * ((p - t) ** 2).sum() / sst, rel=1e-9)


# ---------------------------------------------------------------------------- masking


@pytest.mark.parametrize('which', [0, 1, 2], ids=['y_hat', 'y_true', 'climatology'])
def test_a_nan_in_one_window_drops_that_node_at_that_lead_from_every_window(which):
    fields = list(_fields())
    clean = per_lead_skill_scores(*fields)
    node, lead = 2, 1
    fields[which] = fields[which].copy()
    fields[which][7, node, lead] = np.nan
    out = per_lead_skill_scores(*fields)

    keep = [n for n in range(len(fields[3])) if n != node]
    without_node = per_lead_skill_scores(*[f[:, keep] for f in fields[:3]], fields[3][keep])
    for name in COLUMNS:
        assert np.isfinite(out[name]).all(), name
        assert out[name][lead] == pytest.approx(without_node[name][lead], rel=1e-9), name
        for other in (0, 2):
            assert out[name][other] == pytest.approx(clean[name][other], rel=1e-9), name


def test_a_zero_variance_pixel_is_dropped_from_r_and_kge2_but_counts_in_nse():
    """A constant prediction (and climatology, so the anomaly is constant too) has an
    undefined r and KGE''. The pooled NSE is still defined: its denominator is the truth's."""
    pred, true, clim, weights = _fields()
    node = 2
    pred[:, node] = 3.0
    clim[:, node] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        out = per_lead_skill_scores(pred, true, clim, weights)

    keep = [n for n in range(len(weights)) if n != node]
    without_node = _expected(pred[:, keep], true[:, keep], clim[:, keep], weights[keep])
    with_node = _expected(pred, true, clim, weights)
    for name in COLUMNS:
        assert np.isfinite(out[name]).all(), name
    _assert_scores_equal(out, without_node, columns=[c for c in COLUMNS if 'corr' in c or 'kge2' in c])
    _assert_scores_equal(out, with_node, columns=['nse_latw', 'nse_deseas_latw', 'clim_nse_latw'])


# ------------------------------------------------------------------------ climatology


def test_clim_columns_score_the_climatology_as_the_forecast():
    pred, true, clim, weights = _fields()
    out = per_lead_skill_scores(pred, true, clim, weights)
    as_forecast = per_lead_skill_scores(clim, true, clim, weights)

    for name in ('nse_latw', 'kge2_latw', 'corr_latw'):
        np.testing.assert_allclose(out[f'clim_{name}'], as_forecast[name], rtol=1e-9, err_msg=name)


# ----------------------------------------------------------------------------- inputs


def test_torch_and_numpy_inputs_give_the_same_numbers():
    """The evaluation scripts pass float32 torch tensors; the sums are taken in float64."""
    pred, true, clim, weights = (a.astype(np.float32) for a in _fields())
    from_numpy = per_lead_skill_scores(pred, true, clim, weights)
    from_torch = per_lead_skill_scores(*(torch.from_numpy(a) for a in (pred, true, clim, weights)))

    _assert_scores_equal(from_torch, from_numpy, rtol=0, atol=1e-12)
    as_float64 = [a.astype(np.float64) for a in (pred, true, clim, weights)]
    _assert_scores_equal(from_torch, _expected(*as_float64))


def test_three_windows_and_six_leads_as_in_the_convlstm_harness():
    """Fewer windows than leads: r must not go through `_corr`, which rejects that shape."""
    pred, true, clim, weights = _fields(num_windows=3, num_leads=6)
    out = per_lead_skill_scores(pred, true, clim, weights)

    for values in out.values():
        assert np.asarray(values).shape == (6,)
    _assert_scores_equal(out, _expected(pred, true, clim, weights))
