import json

import pytest
import torch

from mnemos.ablate import budget_table, matched_baseline_config
from mnemos.evaluate import read_ablation
from mnemos.train import Trainer, load_checkpoint, lr_at
from tests.conftest import KINDS, tiny_cfg


@pytest.mark.parametrize("kind", ["none"] + KINDS)
def test_train_smoke(kind, tmp_path):
    cfg = tiny_cfg(kind)
    cfg.train.out_dir = str(tmp_path / kind)
    tr = Trainer(cfg)
    res = tr.train()
    assert torch.isfinite(torch.tensor(res["val_loss"]))
    assert 0.0 <= res["val_acc"] <= 1.0
    lines = [json.loads(l) for l in (tmp_path / kind / "metrics.jsonl").read_text().splitlines()]
    assert any(r["event"] == "train" for r in lines)
    assert (tmp_path / kind / "ckpt.pt").exists()


def test_learning_actually_happens(tmp_path):
    """A tiny model must drive the recall loss below the uniform-guess floor."""
    cfg = tiny_cfg("product_key")
    cfg.train.out_dir = str(tmp_path / "learn")
    cfg.train.steps = 200
    cfg.train.batch_size = 16
    cfg.train.lr = 3e-3
    cfg.train.eval_batches = 4
    tr = Trainer(cfg)
    start = tr.evaluate()["val_loss"]
    end = tr.train()["val_loss"]
    assert end < start - 0.2, f"loss did not move: {start:.3f} -> {end:.3f}"


def test_checkpoint_roundtrip(tmp_path):
    cfg = tiny_cfg("slot")
    cfg.train.out_dir = str(tmp_path / "ck")
    tr = Trainer(cfg)
    tr.train()
    model, loaded_cfg = load_checkpoint(tmp_path / "ck" / "ckpt.pt", "cpu")
    assert loaded_cfg.model.memory.kind == "slot"
    x = torch.randint(0, cfg.model.vocab_size, (2, 16))
    model.eval(), tr.model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x)[0], tr.model.to("cpu")(x)[0], rtol=1e-4, atol=1e-5)


def test_lr_schedule_warms_up_then_decays():
    cfg = tiny_cfg("none").train
    cfg.steps, cfg.warmup, cfg.lr = 100, 10, 1.0
    assert lr_at(0, cfg) < lr_at(9, cfg)
    assert lr_at(9, cfg) == pytest.approx(1.0)
    assert lr_at(99, cfg) < lr_at(50, cfg)
    assert lr_at(99, cfg) >= cfg.lr * cfg.min_lr_frac - 1e-9


@pytest.mark.parametrize("kind", KINDS)
def test_budget_table_reports_a_real_overhead(kind):
    b = budget_table(tiny_cfg(kind))
    assert b["param_overhead_x"] > 1.0
    assert b["params_from_memory"] > 0


@pytest.mark.parametrize("match", ["params", "flops"])
def test_matched_baseline_hits_its_budget(match):
    cfg = tiny_cfg("product_key")
    base, report = matched_baseline_config(cfg, match)
    assert base.model.memory.kind == "none"
    key = "param_ratio" if match == "params" else "flop_ratio"
    assert report[key] >= 1.0
    assert report[key] < 1.15, f"overshot the {match} budget: {report[key]:.3f}"
    assert base.model.d_ff > cfg.model.d_ff


def test_param_matching_a_sparse_memory_overpays_in_flops():
    """Product-key memory is parameter-heavy and FLOP-light, so a parameter-matched
    dense baseline is handed strictly more compute. Documenting that asymmetry is the
    point of reporting both controls."""
    cfg = tiny_cfg("product_key")
    _, p_report = matched_baseline_config(cfg, "params")
    _, f_report = matched_baseline_config(cfg, "flops")
    assert p_report["flop_ratio"] > f_report["flop_ratio"]
    assert p_report["d_ff_baseline"] > f_report["d_ff_baseline"]


def test_read_ablation_reports_both_arms(tmp_path):
    cfg = tiny_cfg("product_key")
    cfg.train.out_dir = str(tmp_path / "ab")
    tr = Trainer(cfg)
    tr.train()
    rep = read_ablation(tr.model, tr.val_data, 2, 8, 123, tr.device)
    assert set(rep) == {"memory_on", "memory_off", "delta_loss", "delta_acc", "diagnostics"}
    assert rep["memory_on"]["scored_tokens"] == rep["memory_off"]["scored_tokens"]
    assert tr.model.memories[0].ablate is False, "ablation flag must be restored"


def test_trainer_rejects_a_corrupt_loss(tmp_path, monkeypatch):
    """A negative cross-entropy means the inputs are corrupt, not that training is
    going well. The loop must stop rather than log it."""
    from mnemos.model import MemoryLM

    cfg = tiny_cfg("none")
    cfg.train.out_dir = str(tmp_path / "guard")
    tr = Trainer(cfg)
    monkeypatch.setattr(
        MemoryLM, "forward",
        lambda self, idx, targets=None, loss_mask=None: (
            torch.zeros(1), (torch.zeros(1, requires_grad=True) - 5.0).mean()
        ),
    )
    with pytest.raises(RuntimeError, match="negative loss"):
        tr.train()


def test_rerunning_into_an_existing_dir_does_not_interleave_metrics(tmp_path):
    """Two runs sharing an out_dir must not stitch their eval curves together."""
    cfg = tiny_cfg("none")
    cfg.train.out_dir = str(tmp_path / "twice")
    cfg.train.eval_every = 2
    Trainer(cfg).train()
    first = (tmp_path / "twice" / "metrics.jsonl").read_text().splitlines()
    Trainer(cfg).train()
    second = (tmp_path / "twice" / "metrics.jsonl").read_text().splitlines()
    assert len(second) == len(first), "second run appended to the first run's log"
    assert (tmp_path / "twice" / "metrics.jsonl.1").exists(), "first run was destroyed"
    starts = [l for l in second if '"start"' in l]
    assert len(starts) == 1


# --- interruption survival ----------------------------------------------------

def _run_cfg(tmp_path, name, steps=8):
    cfg = tiny_cfg("product_key")
    cfg.train.steps = steps
    cfg.train.batch_size = 4
    cfg.train.eval_every = 0
    cfg.train.eval_batches = 1
    cfg.train.ckpt_every = 2
    cfg.train.resume = True
    cfg.train.out_dir = str(tmp_path / name)
    return cfg


def test_resume_reproduces_an_uninterrupted_run(tmp_path, monkeypatch):
    """A killed run picked back up must land exactly where the unbroken run did.

    This is the property an overnight sweep depends on. If the data generator were
    not checkpointed alongside the weights, the resumed run would silently replay
    the batches it already trained on, and the only symptom would be a slightly
    better-looking curve.
    """
    from mnemos.model import MemoryLM

    torch.manual_seed(0)
    reference = Trainer(_run_cfg(tmp_path, "ref")).train()
    ref_weights = {k: v.clone() for k, v in
                   torch.load(tmp_path / "ref" / "ckpt.pt", weights_only=False)["model"].items()}

    original, calls = MemoryLM.forward, {"n": 0}

    def crashing(self, idx, targets=None, loss_mask=None):
        calls["n"] += 1
        if calls["n"] == 5:
            raise RuntimeError("simulated kill")
        return original(self, idx, targets, loss_mask)

    monkeypatch.setattr(MemoryLM, "forward", crashing)
    with pytest.raises(RuntimeError, match="simulated kill"):
        Trainer(_run_cfg(tmp_path, "cut")).train()
    monkeypatch.setattr(MemoryLM, "forward", original)

    assert (tmp_path / "cut" / "ckpt_last.pt").exists(), "no mid-run checkpoint to resume from"
    resumed_trainer = Trainer(_run_cfg(tmp_path, "cut"))
    assert resumed_trainer.start_step == 4, "resumed from the wrong step"
    resumed = resumed_trainer.train()

    assert resumed["val_loss"] == pytest.approx(reference["val_loss"], abs=1e-5)
    got = torch.load(tmp_path / "cut" / "ckpt.pt", weights_only=False)["model"]
    for k, v in ref_weights.items():
        torch.testing.assert_close(got[k], v, rtol=1e-5, atol=1e-6)


def test_resume_is_opt_in(tmp_path):
    cfg = _run_cfg(tmp_path, "optin", steps=4)
    Trainer(cfg).train()
    assert (tmp_path / "optin" / "ckpt_last.pt").exists()
    cfg.train.resume = False
    assert Trainer(cfg).start_step == 0, "resume=False must ignore an existing checkpoint"


def test_rerunning_does_not_interleave_metrics(tmp_path):
    cfg = _run_cfg(tmp_path, "logs", steps=4)
    cfg.train.resume = False
    Trainer(cfg).train()
    first = (tmp_path / "logs" / "metrics.jsonl").read_text()
    Trainer(cfg).train()
    assert (tmp_path / "logs" / "metrics.jsonl.1").read_text() == first
    assert "metrics.jsonl" in [p.name for p in (tmp_path / "logs").iterdir()]


def test_a_corrupt_last_checkpoint_is_survivable(tmp_path, capsys):
    cfg = _run_cfg(tmp_path, "corrupt", steps=4)
    Trainer(cfg).train()
    (tmp_path / "corrupt" / "ckpt_last.pt").write_bytes(b"not a checkpoint")
    tr = Trainer(cfg)
    assert tr.start_step == 0, "an unreadable checkpoint must restart, not crash"
    assert "ignoring unreadable" in capsys.readouterr().out


def test_resume_repairs_a_zero_step_optimizer_state(tmp_path):
    """AdamW divides by 1 - beta1**step, so a restored step of zero is a crash.

    Parameters that go long stretches without a gradient (the kNN key and value
    projections do) leave ragged step counters in a checkpoint, and one that reads
    back as zero killed a sweep run mid-flight.
    """
    cfg = _run_cfg(tmp_path, "zerostep", steps=4)
    tr = Trainer(cfg)
    tr.train()
    ck = torch.load(tmp_path / "zerostep" / "ckpt_last.pt", weights_only=False)
    victim = next(iter(ck["opt"]["state"]))
    step = ck["opt"]["state"][victim]["step"]
    ck["opt"]["state"][victim]["step"] = torch.zeros_like(step) if torch.is_tensor(step) else 0.0
    torch.save(ck, tmp_path / "zerostep" / "ckpt_last.pt")

    cfg.train.steps = 6
    resumed = Trainer(cfg)
    assert all(float(st["step"]) >= 1.0 for st in resumed.opt.state.values())
    resumed.train()          # must not raise ZeroDivisionError
