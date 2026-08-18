import torch
import pytest

from mnemos.registry import MEMORY_REGISTRY, available, build_memory
from tests.conftest import KINDS, PARAMS


@pytest.mark.parametrize("kind", KINDS)
def test_registered(kind):
    assert kind in available() and kind in MEMORY_REGISTRY


@pytest.mark.parametrize("kind", KINDS)
def test_read_shape_and_grad(kind):
    mem = build_memory(kind, d_model=32, **PARAMS[kind])
    mem.reset_memory(2, torch.device("cpu"), torch.float32)
    x = torch.randn(2, 16, 32, requires_grad=True)
    out = mem(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("kind", KINDS)
def test_ablation_zeroes_the_read(kind):
    mem = build_memory(kind, d_model=32, **PARAMS[kind])
    mem.reset_memory(2, torch.device("cpu"), torch.float32)
    x = torch.randn(2, 16, 32)
    mem.ablate = True
    assert torch.equal(mem(x), torch.zeros_like(x))
    assert mem.diagnostics()["ablated"] == 1.0


@pytest.mark.parametrize("kind", KINDS)
def test_accounting_is_populated(kind):
    mem = build_memory(kind, d_model=32, **PARAMS[kind])
    assert mem.memory_param_count() > 0
    assert mem.flops_per_token(64) > 0


def test_product_key_touches_many_slots():
    mem = build_memory("product_key", d_model=32, **PARAMS["product_key"])
    mem(torch.randn(4, 32, 32))
    d = mem.diagnostics()
    assert 0.0 < d["slots_touched_frac"] <= 1.0
    assert d["slot_entropy_bits"] > 0.0


def test_surprise_memory_actually_updates_its_state():
    mem = build_memory("surprise", d_model=32, **PARAMS["surprise"])
    mem.reset_memory(2, torch.device("cpu"), torch.float32)
    assert mem._W.abs().sum() == 0
    mem(torch.randn(2, 32, 32))
    assert mem._W.abs().sum() > 0, "fast weights never moved"


def test_knn_datastore_grows_only_on_commit():
    mem = build_memory("knn", d_model=32, **PARAMS["knn"])
    torch.nn.init.normal_(mem.out.weight)   # undo the zero-init read gate
    x = torch.randn(2, 16, 32)
    out = mem(x)
    assert torch.count_nonzero(out) == 0, "empty datastore must read as zero"
    assert out.requires_grad, "an empty read must still be differentiable"
    assert mem.store_k.shape[0] == 0
    mem.commit()
    assert mem.store_k.shape[0] == 32
    assert torch.count_nonzero(mem(x)) > 0, "a populated datastore must be read"
    assert mem.diagnostics()["store_size"] == 32
    mem.clear_datastore()
    assert mem.store_k.shape[0] == 0


def test_knn_capacity_is_enforced():
    mem = build_memory("knn", d_model=32, capacity=48, d_key=16, topk=4)
    for _ in range(5):
        mem(torch.randn(2, 16, 32))
        mem.commit()
    assert mem.store_k.shape[0] == 48


def test_duplicate_registration_is_rejected():
    from mnemos.registry import register_memory
    with pytest.raises(KeyError):
        register_memory("product_key")(object)
