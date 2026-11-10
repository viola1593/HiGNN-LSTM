"""Finding #20: every `edge_attr` is a constant 1, so the shared `edge_mlp` produces the
same row for every edge. The embedder collapses that to one MLP call broadcast over E.

The collapse is mathematically identical but not bit-exact (a batch-1 `nn.Linear` takes a
different GEMM path than a batch-E one, ~1 ulp), so the all-ones case is checked against the
full MLP at float32 tolerance. The fallback — the moment edges carry real features — must be
bit-exact, since it is the unmodified code path.
"""
import pytest
import torch
from torch_geometric.data import HeteroData

from models.layers.embedder import Embedder

HID_DIM = 64
EDGE_TYPES = [('grid', 'to', 'mesh'), ('ocean', 'to', 'mesh'),
              ('mesh', 'to', 'mesh'), ('mesh', 'to', 'grid')]
N_EDGES = 5000


def make_embedder():
    torch.manual_seed(0)
    return Embedder(grid_node_dim=3, ocean_node_dim=2, mesh_node_dim=4,
                    edge_dim=1, hid_dim=HID_DIM)


def make_data(edge_attr_fn):
    data = HeteroData()
    data['grid'].x = torch.randn(11, 3)
    data['ocean'].x = torch.randn(7, 2)
    data['mesh'].x = torch.randn(5, 4)
    for et in EDGE_TYPES:
        data[et].edge_attr = edge_attr_fn(et)
    return data


def ones(_et):
    return torch.ones(N_EDGES)


def varying(_et):
    return torch.linspace(0.5, 2.0, N_EDGES)


def reference(model, edge_attr):
    """What the pre-#20 embedder computed: the full MLP over every edge."""
    return model.edge_mlp(edge_attr.reshape(-1, 1))


@pytest.fixture
def model():
    return make_embedder()


def test_constant_edges_match_the_full_mlp(model):
    out = model(make_data(ones))
    for et in EDGE_TYPES:
        expected = reference(model, ones(et))
        assert torch.allclose(out[et].edge_attr, expected, atol=1e-5, rtol=1e-4)


def test_constant_edges_give_one_repeated_row(model):
    out = model(make_data(ones))
    for et in EDGE_TYPES:
        rows = out[et].edge_attr
        assert rows.shape == (N_EDGES, HID_DIM)
        assert (rows == rows[0]).all()


def test_non_constant_edges_fall_back_bit_exactly(model):
    out = model(make_data(varying))
    for et in EDGE_TYPES:
        expected = reference(model, varying(et))
        assert torch.equal(out[et].edge_attr, expected)


def edge_mlp_batch_sizes(model, data):
    seen = []
    handle = model.edge_mlp.register_forward_pre_hook(lambda _m, args: seen.append(args[0].shape[0]))
    try:
        model(data)
    finally:
        handle.remove()
    return seen


def test_constant_edges_run_the_mlp_once_per_edge_type(model):
    """The point of #20: ~millions of identical scalars must not go through the MLP."""
    assert edge_mlp_batch_sizes(model, make_data(ones)) == [1] * len(EDGE_TYPES)


def test_non_constant_edges_still_run_the_mlp_over_every_edge(model):
    assert edge_mlp_batch_sizes(model, make_data(varying)) == [N_EDGES] * len(EDGE_TYPES)


def test_no_edges_is_not_a_crash(model):
    out = model(make_data(lambda _et: torch.ones(0)))
    for et in EDGE_TYPES:
        assert out[et].edge_attr.shape == (0, HID_DIM)


def test_gradients_reach_the_edge_mlp(model):
    out = model(make_data(ones))
    sum(out[et].edge_attr.sum() for et in EDGE_TYPES).backward()

    ref = make_embedder()
    total = sum(reference(ref, ones(et)).sum() for et in EDGE_TYPES)
    total.backward()

    for (name, p), (ref_name, ref_p) in zip(model.edge_mlp.named_parameters(),
                                            ref.edge_mlp.named_parameters()):
        assert name == ref_name
        assert p.grad is not None
        assert torch.allclose(p.grad, ref_p.grad, atol=1e-3, rtol=1e-3), name
