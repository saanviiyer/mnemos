"""The sweep driver decides what an overnight job actually computes, so its plan is
worth testing as carefully as the model code."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sweep import SCALES, make_config, plan  # noqa: E402

SMALL_ARMS = ["none", "product_key", "slot", "surprise", "knn"]


def _manifest(runs):
    return [{"name": c.name, "out_dir": c.train.out_dir, "steps": c.train.steps,
             "memory_kind": c.model.memory.kind, "d_ff": c.model.d_ff,
             "seed": c.train.seed} for c in runs]


def test_small_plan_matches_the_launched_sweep():
    """A sweep is already running against this plan. Changing it silently would
    orphan the finished runs and quietly restart them under new directories."""
    want = json.loads((Path(__file__).parent / "fixtures" / "small_manifest.json").read_text())
    got = _manifest(plan("small", [0, 1, 2], SMALL_ARMS, Path("runs/sweep_small"), 10000))
    assert got == want


def test_plan_is_seed_major(tmp_path):
    """Every prefix of the run list must be a complete sweep at some seed count, so
    an interrupted job still yields something comparable."""
    runs = plan("small", [0, 1, 2], SMALL_ARMS, tmp_path, 100)
    per_seed = len(runs) // 3
    assert per_seed == 7
    for block in range(3):
        chunk = runs[block * per_seed : (block + 1) * per_seed]
        assert {c.train.seed for c in chunk} == {block}
        labels = [c.name.split("-")[1] for c in chunk]
        assert labels == SMALL_ARMS + ["matched_params", "matched_flops"]


def test_every_run_owns_a_distinct_directory(tmp_path):
    runs = plan("small", [0, 1, 2], SMALL_ARMS, tmp_path, 100)
    dirs = [c.train.out_dir for c in runs]
    assert len(set(dirs)) == len(dirs)


@pytest.mark.parametrize("scale", sorted(SCALES))
def test_arm_labels_survive_the_summary_parser(scale):
    """summarise() recovers the arm from name.split('-')[1], so a hyphen in a label
    would silently split one arm into two rows."""
    for label in SCALES[scale]["arms"]:
        assert "-" not in label, f"{scale}/{label} would break the summary grouping"
    for base in SCALES[scale].get("controls_for", []):
        assert base in SCALES[scale]["arms"], f"{scale} names a control base that is not an arm"


@pytest.mark.parametrize("scale", sorted(SCALES))
def test_every_arm_builds(scale, tmp_path):
    for arm in SCALES[scale]["arms"]:
        cfg = make_config(scale, arm, 0, tmp_path)
        cfg.validate()
        assert cfg.model.memory.kind == SCALES[scale]["arms"][arm]["kind"]


def test_controls_are_built_per_base_arm(tmp_path):
    """The cluster tier compares several table sizes, so each needs its own matched
    controls and they must not land in the same directory."""
    arms = ["none", "pkm_k32", "pkm_k64"]
    runs = plan("cluster", [0], arms, tmp_path, 100)
    labels = [c.name.split("-")[1] for c in runs]
    assert labels == arms + ["matched_params_pkm_k32", "matched_flops_pkm_k32",
                             "matched_params_pkm_k64", "matched_flops_pkm_k64"]
    assert len({c.train.out_dir for c in runs}) == len(runs)


def test_controls_skip_bases_not_being_run(tmp_path):
    runs = plan("cluster", [0], ["none", "pkm_k32"], tmp_path, 100)
    assert not any("pkm_k64" in c.name for c in runs)


def test_meta_measurement_equals_a_real_build():
    """The bisection counts parameters on the meta device to stay affordable at
    cluster scale. If that ever diverged from a real build, every matched baseline
    would be matched to the wrong budget."""
    from mnemos.ablate import _measure
    from mnemos.model import MemoryLM

    cfg = make_config("small", "product_key", 0, Path("/tmp"), steps=1)
    meta_params, meta_flops = _measure(cfg, cfg.data.seq_len)
    real = MemoryLM(cfg.model)
    assert meta_params == real.param_count()
    assert meta_flops == pytest.approx(real.flops_per_token(cfg.data.seq_len))


def test_cluster_tier_spans_the_table_size_question():
    """The point of the tier: memory parameters must cross from far below the dense
    weights to above them, otherwise the sweep cannot answer what it is asking."""
    from mnemos.ablate import budget_table

    ratios = []
    for arm in ["pkm_k32", "pkm_k512"]:
        bt = budget_table(make_config("cluster", arm, 0, Path("/tmp")))
        ratios.append(bt["params_from_memory"] / bt["params_without_memory"])
    assert ratios[0] < 0.05, "smallest table is not small enough to be a floor"
    assert ratios[1] > 1.0, "largest table never exceeds the dense parameters"


def test_one_failing_run_does_not_abort_the_sweep(tmp_path, monkeypatch, capsys):
    """A 19-hour grid must not be one driver hiccup away from producing nothing.

    An MPS command-buffer fault killed a real sweep at 4 of 21 runs; the remaining
    17 never started. Failures are now recorded and skipped, and because every run
    resumes from its checkpoint, relaunching retries the failed one rather than
    redoing the sweep.
    """
    import scripts.sweep as sweep

    calls = []

    def flaky(cfg):
        calls.append(cfg.name)
        if len(calls) == 2:
            raise RuntimeError("command buffer exited with error status")
        return {"name": cfg.name, "seed": cfg.train.seed, "val_loss": 1.0,
                "val_acc": 0.5, "params_total": 10, "flops_per_token": 10.0}

    monkeypatch.setattr(sweep, "run_one", flaky)
    monkeypatch.setattr(sys, "argv", [
        "sweep.py", "--scale", "small", "--seeds", "0", "--arms", "none,slot",
        "--steps", "1", "--out", str(tmp_path / "s")])
    sweep.main()

    assert len(calls) == 2, "the sweep stopped at the failure instead of continuing"
    out = capsys.readouterr().out
    assert "[FAIL]" in out and "1 run(s) failed" in out
    failures = json.loads((tmp_path / "s" / "failures.json").read_text())
    assert "command buffer" in next(iter(failures.values()))
    assert (tmp_path / "s" / "summary.json").exists(), "surviving runs must still summarise"
