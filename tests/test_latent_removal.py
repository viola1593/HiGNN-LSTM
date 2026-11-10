"""run_latent_removal.py: replace one LSTM input by its train mean in a fixed trained model."""

import pathlib
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import run_latent_removal as rlr
from test_timestep_embedding_required import GNN_EMBEDDING_DIM, HORIZON, N_GRID, SEQ_LEN, make_data, make_model


def test_constants():
    assert rlr.TABLE_SEED == 45
    assert rlr.SEEDS == [str(s) for s in range(42, 50)]
    assert rlr.REMOVALS == {"channel_mean": ["dim1", "dim2", "dim3", "dim4"], "lstm_input": ["twsa", "lead"]}
    assert rlr.STUDY_DIR.endswith("latent_removal_03")


def _output(batch, nodes, steps, channels, value):
    """Outputter output [batch*nodes, steps*channels] with `value(b, n, p, c)` at each position."""
    z = torch.tensor([[[[value(b, n, p, c) for c in range(channels)] for p in range(steps)]
                       for n in range(nodes)] for b in range(batch)], dtype=torch.float32)
    return z.reshape(batch * nodes, -1)


def test_record_means_averages_over_samples_per_lead_cell_step_and_channel():
    hook = rlr.RecordMeans(horizon=2, n_nodes=3, seq_len=4, n_channels=2)
    value = lambda lead, offset: lambda b, n, p, c: 1000 * lead + 100 * n + 10 * p + c + offset * (b + 1)

    # two batches of different size, each with one call per lead: samples at offset 1, 2 and 3
    hook(None, None, _output(2, 3, 4, 2, value(0, 1)))
    hook(None, None, _output(2, 3, 4, 2, value(1, 1)))
    hook(None, None, _output(1, 3, 4, 2, lambda b, n, p, c: 100 * n + 10 * p + c + 3))
    hook(None, None, _output(1, 3, 4, 2, lambda b, n, p, c: 1000 + 100 * n + 10 * p + c + 3))

    mean = hook.cell_mean()
    assert mean.shape == (2, 3, 4, 2)
    for lead in range(2):
        for n in range(3):
            for p in range(4):
                for c in range(2):
                    assert mean[lead, n, p, c] == 1000 * lead + 100 * n + 10 * p + c + 2


def test_recording_leaves_the_output_alone():
    hook = rlr.RecordMeans(horizon=2, n_nodes=N_GRID, seq_len=SEQ_LEN, n_channels=GNN_EMBEDDING_DIM)
    assert hook(None, None, torch.ones(N_GRID, SEQ_LEN * GNN_EMBEDDING_DIM)) is None


def test_channel_mean_is_one_value_per_channel():
    mean = torch.arange(2 * 3 * 4 * 2, dtype=torch.float32).reshape(2, 3, 4, 2)

    fill = rlr.channel_mean(mean)

    assert fill.shape == mean.shape
    assert (fill[..., 0] == 23).all() and (fill[..., 1] == 24).all()


def _lstm_forward(model, *hooks):
    """Forecast, the latents and the LSTM input of every lead, seen after `hooks` as pre-hooks on the LSTM."""
    seen = []
    handles = [model.rnn.register_forward_pre_hook(rlr.on_lstm_input(h)) for h in hooks]
    handles.append(model.rnn.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone())))
    with torch.no_grad():
        forecast, _, latents = model(make_data())
    for handle in handles:
        handle.remove()
    return forecast, torch.stack(latents), torch.stack(seen)  # [leads, nodes, steps, inputs]


def _record(model):
    _, _, inputs = _lstm_forward(model)
    record = rlr.RecordMeans(HORIZON, N_GRID, SEQ_LEN, inputs.shape[-1])
    _lstm_forward(model, record)
    return record


def test_lstm_input_columns_follow_the_order_of_the_concatenation():
    assert rlr.lstm_input_columns(make_model()) == {"lead": [0, 1], "twsa": [2], "dim1": [3], "dim2": [4],
                                                     "dim3": [5], "dim4": [6]}


@pytest.mark.parametrize("skip_gnn", [False, True])
def test_removing_a_latent_channel_only_changes_that_lstm_input(skip_gnn):
    torch.manual_seed(0)
    model = make_model(skip_gnn=skip_gnn).eval()
    forecast, latents, inputs = _lstm_forward(model)
    torch.testing.assert_close(inputs[..., 3:], latents)  # the latent columns are the outputter output
    fill = torch.full((HORIZON, N_GRID, SEQ_LEN, inputs.shape[-1]), 7.0)

    masked_forecast, _, masked = _lstm_forward(model, rlr.ReplaceChannels(fill, rlr.lstm_input_columns(model)["dim2"]))

    assert (masked[..., 4] == 7).all()
    others = [c for c in range(inputs.shape[-1]) if c != 4]
    assert torch.equal(masked[..., others], inputs[..., others])
    assert not torch.isclose(masked_forecast, forecast).all()


def test_a_latent_channel_becomes_its_mean_over_all_leads_cells_and_steps():
    torch.manual_seed(0)
    model = make_model().eval()
    _, latents, inputs = _lstm_forward(model)
    fill = rlr.channel_mean(_record(model).cell_mean())

    _, _, masked = _lstm_forward(model, rlr.ReplaceChannels(fill, rlr.lstm_input_columns(model)["dim2"]))

    torch.testing.assert_close(masked[..., 4], latents[..., 1].mean().expand(HORIZON, N_GRID, SEQ_LEN))
    others = [c for c in range(inputs.shape[-1]) if c != 4]
    assert torch.equal(masked[..., others], inputs[..., others])


def test_the_twsa_history_becomes_one_value_over_all_leads_cells_and_steps():
    torch.manual_seed(0)
    model = make_model(residual_from_last=False).eval()
    forecast, _, inputs = _lstm_forward(model)
    fill = rlr.channel_mean(_record(model).cell_mean())

    masked_forecast, _, masked = _lstm_forward(model, rlr.ReplaceChannels(fill, rlr.lstm_input_columns(model)["twsa"]))

    assert inputs[..., 2].std() > 0
    torch.testing.assert_close(masked[..., 2], inputs[..., 2].mean().expand(HORIZON, N_GRID, SEQ_LEN))
    others = [c for c in range(inputs.shape[-1]) if c != 2]
    assert torch.equal(masked[..., others], inputs[..., others])
    assert not torch.isclose(masked_forecast, forecast).all()


def test_the_lead_embedding_becomes_its_mean_vector_for_every_lead():
    torch.manual_seed(0)
    model = make_model(residual_from_last=False).eval()
    _, _, inputs = _lstm_forward(model)
    fill = rlr.channel_mean(_record(model).cell_mean())
    columns = rlr.lstm_input_columns(model)["lead"]

    _, _, masked = _lstm_forward(model, rlr.ReplaceChannels(fill, columns))

    embeddings = model.timestep_embeddings.weight.detach()
    assert not torch.isclose(embeddings.mean(0)[0], embeddings.mean(0)[1])  # one value per column, not one scalar
    torch.testing.assert_close(masked[..., columns], embeddings.mean(0).expand(HORIZON, N_GRID, SEQ_LEN, -1))
    torch.testing.assert_close(masked[..., 2:], inputs[..., 2:])


def test_every_lead_gets_its_own_fill_over_repeated_batches():
    torch.manual_seed(0)
    model = make_model().eval()
    width = _lstm_forward(model)[2].shape[-1]
    fill = torch.stack([torch.full((N_GRID, SEQ_LEN, width), float(lead)) for lead in range(HORIZON)])
    hook = rlr.ReplaceChannels(fill, rlr.lstm_input_columns(model)["dim1"])

    for _ in range(3):
        _, _, masked = _lstm_forward(model, hook)
        for lead in range(HORIZON):
            assert (masked[lead][..., 3] == lead).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="index_put_ only goes dense on CUDA")
def test_replacing_fits_in_memory_under_deterministic_algorithms():
    """pl `deterministic: True`: an indexed assignment on the full grid tried to allocate 1780 GiB."""
    n_nodes, steps, channels = 20370, 12, 7
    output = torch.ones(2 * n_nodes, steps, channels, device="cuda")
    fill = torch.zeros(1, n_nodes, steps, channels)
    torch.use_deterministic_algorithms(True)
    try:
        replaced = rlr.ReplaceChannels(fill, [4])(None, None, output).reshape(2, n_nodes, steps, channels)
    finally:
        torch.use_deterministic_algorithms(False)

    assert (replaced[..., 4] == 0).all() and (replaced[..., [0, 1, 2, 3, 5, 6]] == 1).all()


def test_replacing_a_single_sample_lstm_input_by_its_mean_changes_nothing():
    torch.manual_seed(0)
    model = make_model(residual_from_last=False).eval()
    forecast, _, inputs = _lstm_forward(model)
    # a whole batch counts as one sample of N_GRID "nodes" in make_data
    record = _record(model)

    replaced, _, _ = _lstm_forward(model, rlr.ReplaceChannels(record.cell_mean(), list(range(inputs.shape[-1]))))

    torch.testing.assert_close(replaced, forecast)
    torch.testing.assert_close(record.cell_mean(), inputs)


def test_rmse_per_lead_is_area_weighted_and_skips_nan_targets():
    pred = torch.zeros(1, 2, 2)
    true = torch.tensor([[[1.0, 3.0], [2.0, float("nan")]]])

    rmse = rlr.rmse_per_lead(pred, true, torch.tensor([3.0, 1.0]))

    assert rmse[0] == pytest.approx(((3 * 1 + 1 * 4) / 4) ** 0.5)
    assert rmse[1] == pytest.approx(3.0)


def test_score_run_scores_every_removal_and_the_unmasked_model_on_test_only(monkeypatch):
    rnn = torch.nn.Identity()
    rnn.input_size = 2 + 1 + 4
    loaders = []

    def predict(model, loader, return_predictions=True):
        loaders.append(loader)
        for _ in range(2):
            rnn(torch.ones(3, 4, rnn.input_size))

    dm = SimpleNamespace(target_seq_len=2, input_seq_len=4, data_train=SimpleNamespace(num_land_nodes=lambda: 3),
                         predict_dataloader=lambda period: period, train_val_test_periods={"train": 0, "val": 1, "test": 2})
    model = SimpleNamespace(model=SimpleNamespace(rnn=rnn, timestep_embed_dim=2, gnn_embedding_dim=4))
    monkeypatch.setattr(rlr, "load_run", lambda run, gpu: (
        SimpleNamespace(predict=predict), dm, model, lambda loader: loaders.append(loader) or torch.tensor([1.0, 2.0])))
    monkeypatch.setattr(rlr, "stored_scores", lambda run: pd.DataFrame({"rmse_latw": [1.0, 2.0]}))

    rows = rlr.score_run("run", "45", 1)

    assert loaders[0] == 0 and set(loaders[1:]) == {2}  # means from train, scores on test
    assert [r[:5] for r in rows[::2]] == [("full", "45", "test", "none", "none")] + [
        ("full", "45", "test", mode, removal) for mode, removals in rlr.REMOVALS.items() for removal in removals]
    assert [r[5] for r in rows] == [0, 1] * 7


def test_unmasked_scores_must_reproduce_the_stored_scores():
    stored = pd.DataFrame({"rmse_latw": [1.402106, 2.034359]})

    rlr.check_reproduces(torch.tensor([1.4022, 2.0343]), stored)
    with pytest.raises(AssertionError, match="stored"):
        rlr.check_reproduces(torch.tensor([1.42, 2.0343]), stored)


def _scores():
    """Two seeds, two leads, test. Seed 46 is 1 cm worse, so unpaired deltas would be wrong."""
    rows = []
    for seed, base in (("45", 2.0), ("46", 3.0)):
        for lead in range(2):
            rows.append(("full", seed, "test", "none", "none", lead, base))
            for i, removal in enumerate(rlr.REMOVALS["channel_mean"]):
                rows.append(("full", seed, "test", "channel_mean", removal, lead, base + 0.01 * (i + 1) * (lead + 1)))
            rows.append(("full", seed, "test", "lstm_input", "twsa", lead, base + 0.4 * (lead + 1)))
            rows.append(("full", seed, "test", "lstm_input", "lead", lead, base + 0.1))
    return pd.DataFrame(rows, columns=rlr.COLUMNS)


def test_lead_deltas_are_paired_to_the_same_checkpoint_in_table_order():
    deltas = rlr.lead_deltas(_scores())

    assert list(deltas.index) == [(s, r) for s in ("45", "46") for r in ("dim1", "dim2", "dim3", "dim4", "twsa", "lead")]
    assert deltas.loc[("46", "dim3")].to_numpy() == pytest.approx([0.03, 0.06])
    assert deltas.loc[("45", "twsa")].to_numpy() == pytest.approx([0.4, 0.8])
    assert deltas.loc[("46", "lead")].to_numpy() == pytest.approx([0.1, 0.1])


def _run_main(tmp_path, monkeypatch, existing):
    calls = {"runs": []}
    monkeypatch.setattr(rlr, "STUDY_DIR", str(tmp_path / "study"))
    monkeypatch.setattr(rlr, "SEED_STUDY", str(tmp_path / "s"))
    monkeypatch.setattr(rlr, "SEEDS", ["45", "46"])
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "x_manifest.json").write_text('{"45": "run45", "46": "run46"}')
    monkeypatch.setattr(sys, "argv", ["run_latent_removal.py", "--gpu", "1"])
    scores = _scores()
    monkeypatch.setattr(rlr, "score_run", lambda run, seed, gpu: calls["runs"].append((run, seed, gpu)) or list(
        scores[scores["seed"] == seed].itertuples(index=False, name=None)))
    monkeypatch.setattr(rlr, "write_tables", lambda scores: calls.update(scores=scores))
    if existing is not None:
        (tmp_path / "study").mkdir()
        existing.to_csv(tmp_path / "study" / "scores.csv", index=False)
    rlr.main()
    return calls


def test_main_scores_every_seed_from_scratch(tmp_path, monkeypatch):
    calls = _run_main(tmp_path, monkeypatch, None)

    assert calls["runs"] == [("run45", "45", 1), ("run46", "46", 1)]
    saved = pd.read_csv(tmp_path / "study" / "scores.csv", dtype={"seed": str})
    assert len(saved) == len(_scores()) == len(calls["scores"])


def test_main_scores_nothing_when_the_scores_are_there(tmp_path, monkeypatch):
    calls = _run_main(tmp_path, monkeypatch, _scores())

    assert calls["runs"] == [] and len(calls["scores"]) == len(_scores())


def test_write_tables_gives_the_table_seed_in_tex_and_every_seed_in_md(tmp_path, monkeypatch):
    monkeypatch.setattr(rlr, "OUT_DIR", str(tmp_path))

    rlr.write_tables(_scores())

    assert sorted(p.name for p in tmp_path.iterdir()) == ["latent_removal.md", "latent_removal_test.tex"]
    tex = (tmp_path / "latent_removal_test.tex").read_text()
    assert "\\label{tab:latent_removal_test}" in tex
    assert "Removed & L1 & L2 & Mean & Mean $|\\Delta|$ \\\\" in tex
    assert "Dim 1 & $+0.010$ & $+0.020$ & $+0.015$ & $0.015$ \\\\" in tex
    assert "TWSA history & $+0.400$ & $+0.800$ & $+0.600$ & $0.600$ \\\\" in tex
    assert "Lead embedding & $+0.100$ & $+0.100$ & $+0.100$ & $0.100$ \\\\" in tex
    assert "2.000 cm" in tex and "3.000" not in tex  # seed 45 only
    for gone in ("Per-cell", "Global mean", "All four", "Sum of"):
        assert gone not in tex

    md = (tmp_path / "latent_removal.md").read_text()
    assert md.count("| Seed | Removed | L1 | L2 | Mean | Mean \\|Δ\\| |") == 1
    assert "| Seed 45 (2.000 cm) | Dim 1 | +0.010 | +0.020 | +0.015 | 0.015 |" in md
    assert "|  | TWSA history | +0.400 | +0.800 | +0.600 | 0.600 |" in md
    assert "| Seed 46 (3.000 cm) | Dim 1 | +0.010 | +0.020 | +0.015 | 0.015 |" in md
    assert md.count("|  | Lead embedding | +0.100 | +0.100 | +0.100 | 0.100 |") == 2
    for gone in ("useful", "distinct", "No MP", "Validation"):
        assert gone not in md


def test_write_tables_mean_absolute_delta_does_not_cancel_signs(tmp_path, monkeypatch):
    monkeypatch.setattr(rlr, "OUT_DIR", str(tmp_path))
    scores = _scores()
    dim1 = (scores["removal"] == "dim1") & (scores["seed"] == "45")
    scores.loc[dim1 & (scores["lead"] == 0), "rmse"] = 2.0 - 0.03
    scores.loc[dim1 & (scores["lead"] == 1), "rmse"] = 2.0 + 0.01

    rlr.write_tables(scores)

    tex = (tmp_path / "latent_removal_test.tex").read_text()
    assert "Dim 1 & $-0.030$ & $+0.010$ & $-0.010$ & $0.020$ \\\\" in tex
    md = (tmp_path / "latent_removal.md").read_text()
    assert "| Seed 45 (2.000 cm) | Dim 1 | -0.030 | +0.010 | -0.010 | 0.020 |" in md
