"""Model flag `residual_from_last`: with True (the default, as before) both models predict a
residual on top of the last input twsa; with False they predict twsa directly, i.e. a zero
network output is a zero (normalized) anomaly = climatology instead of persistence.
The flag is a model kwarg, so old configs/checkpoints keep the base.
"""
import pytest
import torch

from models.pl_convlstm import ConvLSTMModule
from models.pl_hmmodel import HMModule
from test_timestep_embedding_required import (GNN_EMBEDDING_DIM, HIDDEN_DIM, HORIZON, NUM_FEATURES,
                                              NUM_OCEAN_FEATURES, SEQ_LEN, TIMESTEP_EMBED_DIM,
                                              make_data, make_model)


def hignn_pair():
    """the same weights with and without the base"""
    torch.manual_seed(0)
    with_base = make_model()
    without_base = make_model(residual_from_last=False)
    without_base.load_state_dict(with_base.state_dict())
    return with_base, without_base


def test_hignn_default_keeps_the_base():
    assert make_model().residual_from_last


def test_hignn_without_base_differs_by_exactly_the_last_twsa():
    with_base, without_base = hignn_pair()
    data = make_data()
    last_twsa = data['grid'].x[:, -1, 0:1]
    out_with = with_base(data)[0]
    out_without = without_base(data)[0]
    torch.testing.assert_close(out_with - out_without, last_twsa.expand(-1, HORIZON))


def test_hignn_zero_output_proj_is_persistence_or_zero():
    with_base, without_base = hignn_pair()
    for model in (with_base, without_base):
        torch.nn.init.zeros_(model.output_proj.weight)
        torch.nn.init.zeros_(model.output_proj.bias)
    data = make_data()
    torch.testing.assert_close(with_base(data)[0], data['grid'].x[:, -1, 0:1].expand(-1, HORIZON))
    torch.testing.assert_close(without_base(data)[0], torch.zeros(data['grid'].x.shape[0], HORIZON))


@pytest.mark.parametrize('flag', [True, False])
def test_lightning_module_passes_the_flag_through(flag):
    module = HMModule(num_input_features=NUM_FEATURES, num_ocean_features=NUM_OCEAN_FEATURES,
                      sequence_length=SEQ_LEN, output_seq_len=HORIZON,
                      timestep_embed_dim=TIMESTEP_EMBED_DIM, hidden_dim=HIDDEN_DIM,
                      gnn_embedding_dim=GNN_EMBEDDING_DIM, residual_from_last=flag)
    assert module.model.residual_from_last is flag


def make_convlstm(**kw):
    return ConvLSTMModule(input_dim=3, hidden_dim=4, horizon=HORIZON, kernel_size=(3, 3), num_layers=1, **kw)


def test_convlstm_default_keeps_the_base():
    assert make_convlstm().residual_from_last


def test_convlstm_without_base_differs_by_exactly_the_last_twsa():
    torch.manual_seed(0)
    with_base = make_convlstm()
    without_base = make_convlstm(residual_from_last=False)
    without_base.load_state_dict(with_base.state_dict())
    x = torch.randn(2, SEQ_LEN, 3, 5, 6)  # [batch, seq, channels, H, W]
    out_with, out_without = with_base(x), without_base(x)
    assert out_with.shape == (2, HORIZON, 5, 6)
    torch.testing.assert_close(out_with - out_without, x[:, -1:, 0].expand(-1, HORIZON, -1, -1))
