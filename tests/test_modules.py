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


@pytest.mark.parametrize("kind", KINDS)
def test_no_memory_parameter_is_orphaned(kind):
    """Every parameter a memory declares must actually be on the gradient path.

    The kNN module used to carry a k_proj and a v_proj that fed only the datastore,
    whose contents are detached. Nothing trained them for the whole of a run. The
    symptom was not a bad number, it was a ragged optimizer state -- most parameters
    at step 6500, one at step 1 -- which then divided by zero inside AdamW. A
    parameter that never moves is a bug whether or not it crashes.
    """
    import torch.nn as nn

    mem = build_memory(kind, d_model=32, **PARAMS[kind])
    for m in mem.modules():                      # undo zero-initialised read gates
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.05)

    ever_moved = set()
    # two passes with a commit between, so a persistent store is populated for the
    # second one and every code path has run at least once. Per-sequence state is
    # reset between passes exactly as MemoryLM.forward does, or the second backward
    # walks a graph the first one already freed.
    for _ in range(2):
        mem.zero_grad(set_to_none=True)
        if not mem.persistent_state:
            mem.reset_memory(2, torch.device("cpu"), torch.float32)
        mem(torch.randn(2, 64, 32)).sum().backward()
        ever_moved |= {n for n, p in mem.named_parameters()
                       if p.grad is not None and float(p.grad.abs().sum()) > 0}
        if hasattr(mem, "commit"):
            mem.commit()

    dead = [n for n, _ in mem.named_parameters() if n not in ever_moved]
    assert not dead, f"{kind}: parameters trained by nothing: {dead}"


def test_surprise_rates_need_at_least_three_chunks():
    """The momentum and forgetting rates are unlearnable on a short sequence.

    The first chunk multiplies a zero state, so eta and alpha get no gradient there,
    and the last chunk's write is never read back. Two chunks therefore train the
    inner learning rate and leave the other two rates at their initial values, in
    silence. Three chunks is the minimum for the recurrence to be visible to
    autograd at all, and the gradient keeps growing well past that.
    """
    import torch.nn as nn

    def rate_grads(seq_len):
        mem = build_memory("surprise", d_model=32, d_key=16, d_val=16, chunk_size=8)
        for m in mem.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.05)
        mem.reset_memory(2, torch.device("cpu"), torch.float32)
        mem(torch.randn(2, seq_len, 32)).sum().backward()
        return {n: (0.0 if p.grad is None else float(p.grad.abs().sum()))
                for n, p in mem.named_parameters() if n.startswith("raw_")}

    two, eight = rate_grads(16), rate_grads(64)
    assert two["raw_lr"] > 0, "the inner learning rate should train even on two chunks"
    assert two["raw_momentum"] == 0.0 and two["raw_decay"] == 0.0
    assert eight["raw_momentum"] > 0 and eight["raw_decay"] > 0
    assert eight["raw_momentum"] > two["raw_momentum"]
