"""The cached edge indices in a graph directory are keyed only by k, but they also
depend on the mesh size = n_basins + num_ocean_clusters. So `num_ocean_clusters` has to
be part of the meta dict guarding the cache, otherwise changing it from 20 to 40 in the
same variant silently reloads edge indices built for the old mesh.

`_ensure_graph_meta` is exercised on a bare instance so the tests do not open the netCDF
dataset or build any graph.
"""
import json
import os

import pytest

from datamodules.TWSA_heterograph_datamodule import TWSA_HeteroGraphDatamodule

DS_PATH = "data/some_dataset.nc"


def call_ensure(num_ocean_clusters, graph_path="graphs/full_unitsphere_v2"):
    dm = object.__new__(TWSA_HeteroGraphDatamodule)
    dm.graph_variant = 'full'
    dm.graph_embedding_version = 'unitsphere_v2'
    dm.ds_path = DS_PATH
    dm.graph_path = graph_path
    dm.num_ocean_clusters = num_ocean_clusters
    dm._ensure_graph_meta()
    return dm


@pytest.fixture
def in_tmp_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_num_ocean_clusters_is_recorded(in_tmp_cwd):
    dm = call_ensure(20)
    meta = json.loads((in_tmp_cwd / dm.graph_path / 'meta.json').read_text())
    assert meta['num_ocean_clusters'] == 20


def test_same_num_ocean_clusters_reuses_cache(in_tmp_cwd):
    call_ensure(20)
    call_ensure(20)  # must not raise


def test_changed_num_ocean_clusters_rejects_cache(in_tmp_cwd):
    call_ensure(20)
    with pytest.raises(AssertionError, match="num_ocean_clusters"):
        call_ensure(40)


def test_meta_without_num_ocean_clusters_rejects_cache(in_tmp_cwd):
    """A meta.json from before this key was tracked must not pass silently."""
    dm = call_ensure(20)
    meta_file = in_tmp_cwd / dm.graph_path / 'meta.json'
    meta = json.loads(meta_file.read_text())
    del meta['num_ocean_clusters']
    meta_file.write_text(json.dumps(meta))
    with pytest.raises(AssertionError):
        call_ensure(20)
