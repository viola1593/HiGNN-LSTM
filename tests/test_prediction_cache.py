"""get_predictions reuses predictions.pt only when it matches the current prediction dataset."""

import pathlib
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import evaluation_HiGNN  # noqa: E402

NUM_NODES, OUT_LEN, BATCH = 5, 6, 2


def _predictions(num_samples, fill):
    """Batches shaped like trainer.predict output: y_hat and y_true are [B * nodes, T]."""
    batches = []
    for start in range(0, num_samples, BATCH):
        b = min(BATCH, num_samples - start)
        y = torch.full((b * NUM_NODES, OUT_LEN), float(fill))
        batches.append((y, y.clone(), {}))
    return batches


class FakeTrainer:
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.calls = 0

    def predict(self, model, dataloader):
        self.calls += 1
        return _predictions(self.num_samples, fill=1)


def _datamodule(num_samples):
    data_pred = SimpleNamespace(samples=list(range(num_samples)), num_land_nodes=lambda: NUM_NODES)
    dm = SimpleNamespace(data_pred=data_pred)
    dm.GRACE_dataloader = lambda: "dataloader"
    return dm


def _run(tmp_path, monkeypatch, cached_samples, current_samples):
    monkeypatch.setattr(evaluation_HiGNN, "build_model_from_checkpoint", lambda *a, **k: "model")
    cache = tmp_path / "predictions.pt"
    if cached_samples is not None:
        torch.save(_predictions(cached_samples, fill=0), cache)
    trainer = FakeTrainer(current_samples)
    config = {"experiment": {"save_dir": str(tmp_path)}}
    predictions = evaluation_HiGNN.get_predictions(_datamodule(current_samples), trainer, config,
                                                   str(tmp_path), dataset="grace")
    return predictions, trainer, cache


def test_matching_cache_is_reused(tmp_path, monkeypatch):
    predictions, trainer, _ = _run(tmp_path, monkeypatch, cached_samples=3, current_samples=3)
    assert trainer.calls == 0
    assert all((b[0] == 0).all() for b in predictions)


def test_missing_cache_is_predicted_and_saved(tmp_path, monkeypatch):
    predictions, trainer, cache = _run(tmp_path, monkeypatch, cached_samples=None, current_samples=3)
    assert trainer.calls == 1
    assert sum(b[0].shape[0] for b in torch.load(cache)) == 3 * NUM_NODES


def test_save_false_predicts_without_writing_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation_HiGNN, "build_model_from_checkpoint", lambda *a, **k: "model")
    trainer = FakeTrainer(3)
    config = {"experiment": {"save_dir": str(tmp_path)}}

    predictions = evaluation_HiGNN.get_predictions(_datamodule(3), trainer, config, str(tmp_path),
                                                   dataset="grace", save=False)

    assert trainer.calls == 1
    assert sum(b[0].shape[0] for b in predictions) == 3 * NUM_NODES
    assert not (tmp_path / "predictions.pt").exists()


def test_stale_cache_with_other_sample_count_is_predicted_again(tmp_path, monkeypatch):
    # the GRACE case: cache built from the old file (61 samples), current file gives 46
    predictions, trainer, cache = _run(tmp_path, monkeypatch, cached_samples=61, current_samples=46)
    assert trainer.calls == 1
    assert sum(b[0].shape[0] for b in predictions) == 46 * NUM_NODES
    assert all((b[0] == 1).all() for b in predictions)
    assert sum(b[0].shape[0] for b in torch.load(cache)) == 46 * NUM_NODES
