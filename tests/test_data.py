import torch
import pytest

from mnemos.data import build_task
from mnemos.data.synthetic import N_SPECIAL, SEP, TASKS


@pytest.mark.parametrize("kind", sorted(TASKS))
def test_shapes_and_mask_nonempty(kind, gen):
    task = build_task(kind, seq_len=64, vocab_size=64)
    x, y, m = task.batch(8, gen)
    assert x.shape == y.shape == m.shape == (8, 63)
    assert m.dtype == torch.bool and m.any()
    assert x.max() < 64 and x.min() >= 0


def test_associative_recall_answers_are_recoverable(gen):
    """The scored positions must have a unique correct answer present earlier."""
    task = build_task("associative_recall", seq_len=64, vocab_size=64,
                      n_pairs=8, n_queries=2)
    x, y, m = task.batch(16, gen)
    for b in range(x.shape[0]):
        body_end = 1 + 2 * task.n_pairs
        keys = x[b, 1:body_end:2]
        vals = x[b, 2:body_end + 1:2]
        assert len(set(keys.tolist())) == len(keys), "keys must be distinct within a sequence"
        for t in m[b].nonzero().flatten().tolist():
            query_key = x[b, t]
            hit = (keys == query_key).nonzero().flatten()
            assert hit.numel() == 1, "query key must appear exactly once in the body"
            assert vals[hit.item()] == y[b, t], "target is not the value stored for that key"


def test_associative_recall_rejects_impossible_shapes():
    with pytest.raises(ValueError):
        build_task("associative_recall", seq_len=8, vocab_size=64, n_pairs=8, n_queries=2)


def test_copy_mask_covers_the_copy_span(gen):
    task = build_task("copy", seq_len=32, vocab_size=64, span=12)
    x, y, m = task.batch(4, gen)
    assert m.sum(dim=1).unique().tolist() == [12]
    assert (x[:, 13] == SEP).all()


def test_needle_answer_matches_planted_fact(gen):
    task = build_task("needle", seq_len=48, vocab_size=64, n_facts=1)
    x, y, m = task.batch(8, gen)
    for b in range(x.shape[0]):
        t = m[b].nonzero().flatten().item()
        key = x[b, t]
        planted = (x[b, :t - 1] == key).nonzero().flatten()
        assert planted.numel() >= 1
        assert x[b, planted[0] + 1] == y[b, t]


def test_unknown_task_raises():
    with pytest.raises(KeyError):
        build_task("does_not_exist", seq_len=32, vocab_size=64)


# --- byte-level text ---------------------------------------------------------

def _corpus(tmp_path, n=4000):
    p = tmp_path / "corpus.txt"
    p.write_bytes(bytes((i * 7 + 13) % 256 for i in range(n)))
    return p


def test_text_dataset_shapes_and_next_token_alignment(tmp_path, gen):
    from mnemos.data import ByteTextDataset
    ds = ByteTextDataset(_corpus(tmp_path), seq_len=32)
    x, y, m = ds.batch(4, gen)
    assert x.shape == y.shape == (4, 32)
    assert m.all(), "text is scored at every position"
    assert x.max() < ByteTextDataset.vocab_size
    torch.testing.assert_close(x[:, 1:], y[:, :-1])


def test_text_split_is_disjoint(tmp_path):
    from mnemos.data import ByteTextDataset
    p = _corpus(tmp_path)
    tr = ByteTextDataset(p, seq_len=16, split="train", val_frac=0.25)
    va = ByteTextDataset(p, seq_len=16, split="val", val_frac=0.25)
    assert tr.data.numel() + va.data.numel() == 4000
    assert abs(va.data.numel() - 1000) <= 1


def test_text_dataset_errors_are_actionable(tmp_path):
    from mnemos.data import ByteTextDataset
    with pytest.raises(FileNotFoundError, match="synthetic"):
        ByteTextDataset(tmp_path / "nope.txt", seq_len=8)
    small = tmp_path / "small.txt"
    small.write_bytes(b"abc")
    with pytest.raises(ValueError, match="need >"):
        ByteTextDataset(small, seq_len=64)


def test_text_requires_a_path():
    from mnemos.config import DataConfig
    from mnemos.data import build_dataset
    with pytest.raises(ValueError, match="data.path"):
        build_dataset(DataConfig(kind="text"))


def test_text_trains_end_to_end(tmp_path):
    from mnemos.config import ExperimentConfig
    from mnemos.train import Trainer
    cfg = ExperimentConfig.from_dict({
        "model": {"vocab_size": 256, "d_model": 32, "n_layers": 2, "n_heads": 2,
                  "d_ff": 64, "max_seq_len": 32,
                  "memory": {"kind": "knn", "layers": [1],
                             "params": {"capacity": 256, "d_key": 16, "topk": 4}}},
        "data": {"kind": "text", "seq_len": 32, "path": str(_corpus(tmp_path))},
        "train": {"steps": 3, "batch_size": 4, "eval_every": 0, "eval_batches": 1,
                  "device": "cpu", "out_dir": str(tmp_path / "text")},
    })
    res = Trainer(cfg).train()
    assert res["val_loss"] > 0
