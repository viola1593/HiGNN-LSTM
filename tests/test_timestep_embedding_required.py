"""Finding #10: `timestep_embed_dim=None` was the documented default but never worked --
the model unconditionally uses the embedding, so `None` died with
`TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'` while building the LSTM.
The option is now gone: the dimension is a required argument and `None` is rejected up front.
"""
import pytest
import torch
from torch_geometric.data import HeteroData

from models.hierarchical_model import HiGNN_LSTM
from models.pl_hmmodel import HMModule

NUM_FEATURES = 3
NUM_OCEAN_FEATURES = 2
SEQ_LEN = 4
HORIZON = 2
HIDDEN_DIM = 8
GNN_EMBEDDING_DIM = 4
TIMESTEP_EMBED_DIM = 2

N_GRID, N_OCEAN, N_MESH = 6, 3, 4


def make_model(**overrides):
    kwargs = dict(num_features=NUM_FEATURES, num_ocean_features=NUM_OCEAN_FEATURES,
                  horizon=HORIZON, timestep_embed_dim=TIMESTEP_EMBED_DIM,
                  sequence_length=SEQ_LEN, hidden_dim=HIDDEN_DIM,
                  num_processing_gnns=2, gnn_embedding_dim=GNN_EMBEDDING_DIM)
    kwargs.update(overrides)
    return HiGNN_LSTM(**kwargs)


def make_data():
    torch.manual_seed(0)
    data = HeteroData()
    data['grid'].x = torch.randn(N_GRID, SEQ_LEN, NUM_FEATURES)
    data['ocean'].x = torch.randn(N_OCEAN, SEQ_LEN, NUM_OCEAN_FEATURES)
    data['mesh'].x = torch.randn(N_MESH, 3)
    for et, (n_src, n_dst) in {('grid', 'to', 'mesh'): (N_GRID, N_MESH),
                               ('ocean', 'to', 'mesh'): (N_OCEAN, N_MESH),
                               ('mesh', 'to', 'mesh'): (N_MESH, N_MESH),
                               ('mesh', 'to', 'grid'): (N_MESH, N_GRID)}.items():
        n_edges = max(n_src, n_dst)
        data[et].edge_index = torch.stack((torch.arange(n_edges) % n_src,
                                           torch.arange(n_edges) % n_dst))
        data[et].edge_attr = torch.ones(n_edges)
    return data


def test_dimension_is_required():
    with pytest.raises(TypeError, match='timestep_embed_dim'):
        HiGNN_LSTM(num_features=NUM_FEATURES, num_ocean_features=NUM_OCEAN_FEATURES,
                   horizon=HORIZON)


def test_none_is_rejected_with_a_readable_error():
    with pytest.raises(ValueError, match='timestep_embed_dim'):
        make_model(timestep_embed_dim=None)


def test_lightning_module_requires_the_dimension_too():
    with pytest.raises(TypeError, match='timestep_embed_dim'):
        HMModule(num_input_features=NUM_FEATURES, num_ocean_features=NUM_OCEAN_FEATURES,
                 sequence_length=SEQ_LEN, output_seq_len=HORIZON)


def test_lightning_module_passes_the_dimension_through():
    module = HMModule(num_input_features=NUM_FEATURES, num_ocean_features=NUM_OCEAN_FEATURES,
                      sequence_length=SEQ_LEN, output_seq_len=HORIZON,
                      timestep_embed_dim=TIMESTEP_EMBED_DIM, hidden_dim=HIDDEN_DIM,
                      gnn_embedding_dim=GNN_EMBEDDING_DIM)
    assert module.model.timestep_embeddings.embedding_dim == TIMESTEP_EMBED_DIM


def test_the_embedding_feeds_the_lstm():
    model = make_model()
    assert model.timestep_embeddings.num_embeddings == HORIZON
    assert model.rnn.input_size == TIMESTEP_EMBED_DIM + 1 + GNN_EMBEDDING_DIM


def test_shared_forward_always_returns_film_parameters():
    model = make_model()
    data = make_data()
    data['grid'].x = data['grid'].x.view(N_GRID, -1)
    data['ocean'].x = data['ocean'].x.view(N_OCEAN, -1)
    embedded = model.grid2mesh(model.embedder(data))

    embed = model.timestep_embeddings(torch.tensor(0))
    _, *film = model.shared_forward(embedded, embed)
    assert len(film) == 4
    assert all(p is not None for p in film)


def test_forward_runs():
    model = make_model()
    forecast, m2m_graphs, gnn_outputs = model(make_data())
    assert forecast.shape == (N_GRID, HORIZON)
    assert len(m2m_graphs) == HORIZON
    assert len(gnn_outputs) == HORIZON
