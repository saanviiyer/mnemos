import pytest

from mnemos.config import ExperimentConfig
from tests.conftest import tiny_cfg


def test_roundtrip(tmp_path):
    cfg = tiny_cfg("product_key")
    p = tmp_path / "c.yaml"
    cfg.save(p)
    again = ExperimentConfig.load(p)
    assert again.to_dict() == cfg.to_dict()


def test_shipped_configs_are_valid():
    from pathlib import Path
    for p in sorted(Path("configs").glob("*.yaml")):
        ExperimentConfig.load(p)


@pytest.mark.parametrize("bad", [
    {"model": {"d_model": 30, "n_heads": 4}},
    {"model": {"n_layers": 2, "memory": {"kind": "slot", "layers": [5]}}},
    {"model": {"memory": {"kind": "slot", "layers": []}}},
    {"model": {"memory": {"kind": "none", "layers": [0]}}},
    {"model": {"memory": {"kind": "slot", "layers": [0], "placement": "sideways"}}},
    {"model": {"max_seq_len": 16}, "data": {"seq_len": 64}},
])
def test_invalid_configs_are_rejected(bad):
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict(bad)


def test_unknown_memory_kind_fails_at_build():
    from mnemos.model import MemoryLM
    cfg = ExperimentConfig.from_dict(
        {"model": {"memory": {"kind": "telepathy", "layers": [0]}}}
    )
    with pytest.raises(KeyError):
        MemoryLM(cfg.model)


def test_surprise_config_rejects_too_few_chunks():
    from mnemos.config import ExperimentConfig
    base = tiny_cfg("surprise")
    d = base.to_dict()
    d["model"]["memory"]["params"]["chunk_size"] = 32     # 64 // 32 == 2 chunks
    with pytest.raises(ValueError, match="unlearnable below 3 chunks"):
        ExperimentConfig.from_dict(d)
    d["model"]["memory"]["params"]["chunk_size"] = 8      # 8 chunks
    ExperimentConfig.from_dict(d)
