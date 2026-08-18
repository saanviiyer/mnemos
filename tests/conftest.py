import torch
import pytest

from mnemos.config import ExperimentConfig

KINDS = ["product_key", "slot", "surprise", "knn"]

PARAMS = {
    "product_key": {"n_keys": 16, "n_heads": 2, "topk": 4, "half_dim": 16, "v_dim": 32},
    "slot": {"n_slots": 8, "n_heads": 2, "chunk_size": 8},
    "surprise": {"d_key": 16, "d_val": 16, "chunk_size": 8},
    "knn": {"capacity": 512, "d_key": 16, "topk": 4},
}


def tiny_cfg(kind="none", layers=None, placement="parallel", **over) -> ExperimentConfig:
    d = {
        "name": f"test-{kind}",
        "model": {
            "vocab_size": 64, "d_model": 32, "n_layers": 2, "n_heads": 2,
            "d_ff": 64, "max_seq_len": 64,
            "memory": {
                "kind": kind,
                "layers": layers if layers is not None else ([1] if kind != "none" else []),
                "placement": placement,
                "params": PARAMS.get(kind, {}),
            },
        },
        "data": {"kind": "associative_recall", "seq_len": 64, "vocab_size": 64,
                 "params": {"n_pairs": 8, "n_queries": 2}},
        "train": {"steps": 4, "batch_size": 4, "eval_every": 0, "eval_batches": 2,
                  "log_every": 100, "device": "cpu"},
    }
    cfg = ExperimentConfig.from_dict(d)
    for k, v in over.items():
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = getattr(node, p)
        setattr(node, parts[-1], v)
    return cfg.validate()


@pytest.fixture
def gen():
    return torch.Generator().manual_seed(0)
