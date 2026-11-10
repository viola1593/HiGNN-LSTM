"""The ocean clustering is geometric: k-means over a unit-sphere embedding of the
pixel coordinates, with the pixel features used only for the per-cluster aggregation.

These tests pin that contract down, because the signature used to advertise an `sst`
default that the call site silently overrode. `cluster_pixels` is exercised on a bare
instance so the tests do not pull in normalization, graph construction or the
hydrobasin cache.
"""
import pickle

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from datamodules.TWSA_heterograph_dataset import TWSAGraphDataset

NUM_CLUSTERS = 3
NUM_TIMESTEPS = 5
VAR_LIST = ['twsa', 'sst', 't2m']

# three regions far enough apart that the correct 3-means partition is unambiguous,
# and spread over the antimeridian so a naive lon/lat clustering would split a blob
BLOB_CENTRES = [(0.0, 0.0), (120.0, 10.0), (-120.0, 40.0)]


def make_ocean(seed=0, feature_offset=0.0):
    """A synthetic ocean: three well-separated blobs of pixels.

    Returns (coords, features) with coords (num_pixels, 2) as [lon, lat] and
    features (num_vars, num_timesteps, num_pixels), plus the blob membership.
    """
    rng = np.random.default_rng(seed)
    coords, blob_id = [], []
    for b, (lon, lat) in enumerate(BLOB_CENTRES):
        for dlon in (-1.5, -0.5, 0.5, 1.5):
            for dlat in (-1.5, -0.5, 0.5, 1.5):
                coords.append((lon + dlon, lat + dlat))
                blob_id.append(b)
    coords = np.array(coords, dtype=np.float64)
    blob_id = np.array(blob_id)

    features = rng.normal(size=(len(VAR_LIST), NUM_TIMESTEPS, len(coords))) + feature_offset
    return coords, features, blob_id


def build(graph_path, ocean_clustering=None):
    obj = TWSAGraphDataset.__new__(TWSAGraphDataset)
    obj.ocean_clustering = ocean_clustering
    obj.num_ocean_clusters = NUM_CLUSTERS
    obj.graph_path = str(graph_path)
    obj.train_val_test_data = 'train'
    obj.input_vars = VAR_LIST
    graph_path.mkdir(parents=True, exist_ok=True)
    return obj


def cluster(obj, coords, features):
    return obj.cluster_pixels((coords, features), var_list=VAR_LIST,
                              num_timesteps=NUM_TIMESTEPS, pixel_type='ocean')


# --- the embedding the clustering runs on ------------------------------------------

def test_unit_sphere_embedding_matches_the_spherical_formula():
    obj = TWSAGraphDataset.__new__(TWSAGraphDataset)
    coords = np.array([[0.0, 0.0], [90.0, 0.0], [0.0, 90.0], [-120.0, -45.0]])

    emb = obj.unit_sphere_embedding(coords)

    lon, lat = np.deg2rad(coords[:, 0]), np.deg2rad(coords[:, 1])
    expected = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)
    np.testing.assert_allclose(emb, expected, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-12)


def test_unit_sphere_embedding_is_continuous_across_the_antimeridian():
    obj = TWSAGraphDataset.__new__(TWSAGraphDataset)
    emb = obj.unit_sphere_embedding(np.array([[179.9, 0.0], [-179.9, 0.0], [0.0, 0.0]]))

    across_the_seam = np.linalg.norm(emb[0] - emb[1])
    half_the_globe = np.linalg.norm(emb[0] - emb[2])
    assert across_the_seam < 1e-2, "neighbouring pixels must not be split by the lon wrap"
    assert half_the_globe > 1.0


def test_clustering_inputs_are_the_coordinate_embedding_alone():
    """What `prepare_inputs_clustering` builds must be what its docstring claims:
    the lon/lat embedding, with no pixel variables mixed in."""
    obj = TWSAGraphDataset.__new__(TWSAGraphDataset)
    obj.input_vars = VAR_LIST
    coords, features, _ = make_ocean()

    k_means_input = obj.prepare_inputs_clustering((coords, features))

    assert k_means_input.shape == (len(coords), 3)
    np.testing.assert_allclose(k_means_input, obj.unit_sphere_embedding(coords), atol=1e-12)


# --- the clustering itself -----------------------------------------------------------

def test_clusters_recover_well_separated_regions(tmp_path):
    obj = build(tmp_path)
    coords, features, blob_id = make_ocean()

    cluster(obj, coords, features)

    labels = np.asarray(obj.ocean_clustering)
    assert labels.shape == (len(coords),)
    assert set(np.unique(labels)) == set(range(NUM_CLUSTERS))
    for b in range(len(BLOB_CENTRES)):
        assert len(np.unique(labels[blob_id == b])) == 1, f"blob {b} was split across clusters"
    assert len(np.unique([labels[blob_id == b][0] for b in range(len(BLOB_CENTRES))])) == NUM_CLUSTERS


def test_clustering_ignores_the_pixel_features(tmp_path):
    """The labels and the cluster geometry depend on the coordinates only. This is the
    documented behaviour, not an accident of how `cluster_pixels` is called."""
    coords, features, _ = make_ocean()
    _, other_features, _ = make_ocean(seed=99, feature_offset=50.0)

    # separate graph_paths so the second run refits instead of loading the first model
    obj_a = build(tmp_path / 'a')
    coords_a, _ = cluster(obj_a, coords, features)
    obj_b = build(tmp_path / 'b')
    coords_b, _ = cluster(obj_b, coords, other_features)

    np.testing.assert_array_equal(np.asarray(obj_a.ocean_clustering),
                                  np.asarray(obj_b.ocean_clustering))
    np.testing.assert_allclose(coords_a, coords_b, atol=1e-12)


def test_output_shapes(tmp_path):
    obj = build(tmp_path)
    coords, features, _ = make_ocean()

    coarse_coords, coarse_features = cluster(obj, coords, features)

    assert coarse_coords.shape == (NUM_CLUSTERS, 2)
    assert coarse_features.shape == (len(VAR_LIST), NUM_TIMESTEPS, NUM_CLUSTERS)


def test_coarsened_coords_are_the_spherical_means_of_the_clusters(tmp_path):
    """The node position is the mean of the members' unit-sphere xyz, rescaled onto the
    sphere - not the mean of their lon/lat."""
    obj = build(tmp_path)
    coords, features, _ = make_ocean()

    coarse_coords, _ = cluster(obj, coords, features)

    labels = np.asarray(obj.ocean_clustering)
    for c in range(NUM_CLUSTERS):
        mean_xyz = obj.unit_sphere_embedding(coords[labels == c]).mean(axis=0)
        np.testing.assert_allclose(obj.unit_sphere_embedding(coarse_coords[c][None])[0],
                                   mean_xyz / np.linalg.norm(mean_xyz), atol=1e-12)


def test_a_cluster_across_the_antimeridian_is_placed_on_the_antimeridian(tmp_path):
    """The mean of lon -179.5 and 179.5 is 0, on the other side of the globe. The node of a
    cluster straddling the dateline has to sit on the dateline, next to its members."""
    coords = np.array([[179.5, 0.0], [-179.5, 0.0], [179.5, 1.0], [-179.5, 1.0],
                       [0.0, 0.0], [0.0, 1.0], [90.0, 0.0], [90.0, 1.0]])
    labels = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    features = np.zeros((len(VAR_LIST), NUM_TIMESTEPS, len(coords)))

    coarse_coords, _ = cluster(build(tmp_path, ocean_clustering=labels), coords, features)

    lon, lat = coarse_coords[0]
    assert abs(abs(lon) - 180.0) < 1e-9
    assert abs(lat - 0.5) < 1e-3


def test_coarsened_features_are_the_cluster_means(tmp_path):
    obj = build(tmp_path)
    coords, features, _ = make_ocean()

    _, coarse_features = cluster(obj, coords, features)

    labels = np.asarray(obj.ocean_clustering)
    for c in range(NUM_CLUSTERS):
        np.testing.assert_allclose(coarse_features[:, :, c], features[:, :, labels == c].mean(axis=-1),
                                   rtol=1e-6)


def test_clustering_is_deterministic(tmp_path):
    coords, features, _ = make_ocean()

    first = build(tmp_path / 'first')
    cluster(first, coords, features)
    second = build(tmp_path / 'second')
    cluster(second, coords, features)

    np.testing.assert_array_equal(np.asarray(first.ocean_clustering),
                                  np.asarray(second.ocean_clustering))


# --- reuse of a fitted clustering ----------------------------------------------------

def test_a_predefined_clustering_is_used_unchanged(tmp_path):
    """The val/test datasets are handed the training labels and must not refit."""
    coords, features, _ = make_ocean()
    train = build(tmp_path / 'train')
    cluster(train, coords, features)
    train_labels = np.asarray(train.ocean_clustering)

    val = build(tmp_path / 'val', ocean_clustering=train_labels)
    coarse_coords, coarse_features = cluster(val, coords, features)

    np.testing.assert_array_equal(np.asarray(val.ocean_clustering), train_labels)
    for c in range(NUM_CLUSTERS):
        np.testing.assert_allclose(coarse_features[:, :, c],
                                   features[:, :, train_labels == c].mean(axis=-1), rtol=1e-6)
    assert not (tmp_path / 'val' / f'ocean_clustering_model_{NUM_CLUSTERS}.pkl').exists()


def test_a_cached_model_reproduces_the_same_labels(tmp_path):
    coords, features, _ = make_ocean()

    fitted = build(tmp_path)
    cluster(fitted, coords, features)
    assert (tmp_path / f'ocean_clustering_model_{NUM_CLUSTERS}.pkl').exists()

    reloaded = build(tmp_path)
    cluster(reloaded, coords, features)

    np.testing.assert_array_equal(np.asarray(reloaded.ocean_clustering),
                                  np.asarray(fitted.ocean_clustering))


def test_an_unloadable_cached_model_is_not_silently_refit(tmp_path, monkeypatch):
    """Only a missing cache file may trigger a refit; any other load error must surface
    instead of the cached model being quietly replaced."""
    coords, features, _ = make_ocean()
    cluster(build(tmp_path), coords, features)

    def broken_load(f):
        raise TypeError("incompatible pickle")
    monkeypatch.setattr(pickle, "load", broken_load)

    with pytest.raises(TypeError, match="incompatible pickle"):
        cluster(build(tmp_path), coords, features)


# --- misuse is rejected loudly -------------------------------------------------------

@pytest.mark.parametrize("predefined", [False, True])
def test_a_missing_graph_path_is_rejected(tmp_path, predefined):
    """Without a graph_path there is nowhere to cache the model or save the cluster plot."""
    coords, features, _ = make_ocean()
    labels = np.resize(np.arange(NUM_CLUSTERS), len(coords)) if predefined else None
    obj = build(tmp_path, ocean_clustering=labels)
    obj.graph_path = None

    with pytest.raises(AssertionError, match="graph_path"):
        cluster(obj, coords, features)


@pytest.mark.parametrize("delta", [-1, 1])
def test_a_predefined_clustering_must_label_every_pixel(tmp_path, delta):
    coords, features, _ = make_ocean()
    labels = np.resize(np.arange(NUM_CLUSTERS), len(coords) + delta)
    obj = build(tmp_path, ocean_clustering=labels)

    with pytest.raises(AssertionError, match="per ocean pixel"):
        cluster(obj, coords, features)
