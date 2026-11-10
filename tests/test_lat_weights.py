"""cos(lat) area weighting in the evaluation metrics: the weighted `_rmse` and the
area-weighted basin means. The *_latw columns the scripts write are covered in
test_c2a (compare_models) and test_c3a (evaluation_ConvLSTM)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utils.eval_metrics import _rmse, basin_wide_mean_twsa, get_basin_wide_mean_twsa

EQUATOR_AND_60 = torch.cos(torch.deg2rad(torch.tensor([0.0, 60.0])))  # [1, 0.5]


# ---------------------------------------------------------------- weighted _rmse


def test_unit_weights_give_the_unweighted_rmse():
    gen = torch.Generator().manual_seed(0)
    y_hat, y_true = torch.randn(5, 7, 6, generator=gen), torch.randn(5, 7, 6, generator=gen)
    torch.testing.assert_close(_rmse(y_hat, y_true, lat_weights=torch.ones(7)), _rmse(y_hat, y_true))


@pytest.mark.parametrize("shape", [(3, 2), (4, 2, 6)])  # [leads, N] ConvLSTM, [windows, N, leads] HiGNN
def test_weights_act_along_the_node_axis(shape):
    """Error 0 at the equator and 1 at 60°: the 60° cell carries half the weight."""
    y_true = torch.zeros(shape)
    y_hat = y_true.clone()
    y_hat[:, 1] = 1.0
    out = _rmse(y_hat, y_true, lat_weights=EQUATOR_AND_60)
    assert out.shape == (shape[0],) + shape[2:]
    torch.testing.assert_close(out, torch.full_like(out, (0.5 / 1.5) ** 0.5))


def test_weighted_rmse_is_normalised_by_the_kept_weight():
    """A constant error of 3 is an RMSE of 3 however much the mask drops: the
    weights need not sum to 1 and masking does not bias the score low."""
    y_true = torch.zeros(4, 6, 2)
    mask = torch.ones_like(y_true, dtype=torch.bool)
    mask[:, :4] = False
    weights = torch.cos(torch.deg2rad(torch.linspace(-50, 70, 6)))
    out = _rmse(y_true + 3.0, y_true, mask=mask, lat_weights=weights)
    torch.testing.assert_close(out, torch.full((4, 2), 3.0))


def test_weighted_rmse_drops_nan_cells_with_their_weight():
    y_hat = torch.tensor([[0.0, 1.0, 5.0], [0.0, 1.0, 5.0]])
    y_true = torch.tensor([[0.0, 0.0, float("nan")], [0.0, 0.0, float("nan")]])
    weights = torch.tensor([1.0, 0.5, 0.9])
    torch.testing.assert_close(_rmse(y_hat, y_true, lat_weights=weights),
                               torch.full((2,), (0.5 / 1.5) ** 0.5))


# ---------------------------------------------------------------- basin means


class _Datamodule:
    """What basin_grouping reads: basins 1 and 2, plus an Antarctica node it drops."""

    data_pred = SimpleNamespace(land_to_basin_ids=np.array([1, 1, 2, 2, 3]))

    def get_antarctica_landmask(self):
        return np.array([False, False, False, False, True])


BASIN_WEIGHTS = torch.cos(torch.deg2rad(torch.tensor([0.0, 60.0, 0.0, 60.0])))  # kept nodes


def _field(num_steps=6):
    """[T, N] on the kept nodes.

    basin 1: 1 at the equator, 4 at 60° -> weighted (1 + 0.5*4) / 1.5 = 2, plain 2.5
    basin 2: 2 at the equator, NaN at 60° -> 2 either way, the NaN drops its weight
    """
    return np.tile(np.array([1.0, 4.0, 2.0, np.nan], dtype="float32"), (num_steps, 1))


def test_basin_mean_weights_the_cells_by_cos_lat():
    field = _field()
    pred, true, clim, ids = basin_wide_mean_twsa(field, field, field, _Datamodule(),
                                                 lat_weights=BASIN_WEIGHTS)
    assert list(ids) == [1, 2]
    for out in (pred, true, clim):
        np.testing.assert_allclose(out, np.tile([2.0, 2.0], (6, 1)), rtol=1e-6)


def test_basin_mean_without_weights_is_the_plain_mean():
    field = _field()
    pred, _, _, _ = basin_wide_mean_twsa(field, field, field, _Datamodule())
    np.testing.assert_allclose(pred, np.tile([2.5, 2.0], (6, 1)), rtol=1e-6)


def test_get_basin_wide_mean_twsa_passes_the_weights_on():
    num_windows, num_leads = 3, 6
    twsa = torch.tensor(_field(num_leads)).T[None].repeat(num_windows, 1, 1)  # [B, N, T]
    climatology = np.tile(_field(1), (12, 1))                                  # [12, N]
    months = torch.zeros(num_windows, 4, num_leads, dtype=torch.long)          # [B, N, T]
    pred, true, clim, _, _ = get_basin_wide_mean_twsa(twsa, twsa, climatology, months,
                                                      _Datamodule(), lat_weights=BASIN_WEIGHTS)
    for out in (pred, true, clim):
        np.testing.assert_allclose(out, np.full((num_windows, num_leads, 2), 2.0), rtol=1e-6)
