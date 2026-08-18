import torch
import pytest

from mnemos.model import MemoryLM
from tests.conftest import KINDS, tiny_cfg


@pytest.mark.parametrize("kind", ["none"] + KINDS)
def test_forward_and_loss(kind):
    cfg = tiny_cfg(kind)
    m = MemoryLM(cfg.model)
    x = torch.randint(0, cfg.model.vocab_size, (3, 32))
    y = torch.randint(0, cfg.model.vocab_size, (3, 32))
    logits, loss = m(x, y)
    assert logits.shape == (3, 32, cfg.model.vocab_size)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("kind", ["none"] + KINDS)
def test_no_future_leakage(kind):
    """Perturbing the last token must not change any earlier position's logits.

    This is the test that catches a memory module writing before it reads. Every
    stateful module here is chunked precisely so this passes.
    """
    cfg = tiny_cfg(kind)
    m = MemoryLM(cfg.model).eval()
    x = torch.randint(3, cfg.model.vocab_size, (2, 24))
    with torch.no_grad():
        m.reset_memory(2, force=True)
        a, _ = m(x)
        x2 = x.clone()
        x2[:, -1] = (x2[:, -1] + 7) % cfg.model.vocab_size
        m.reset_memory(2, force=True)
        b, _ = m(x2)
    assert not torch.equal(x, x2)
    torch.testing.assert_close(a[:, :-1], b[:, :-1], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_mid_sequence_perturbation_is_causal(kind):
    cfg = tiny_cfg(kind)
    m = MemoryLM(cfg.model).eval()
    x = torch.randint(3, cfg.model.vocab_size, (2, 24))
    cut = 13
    with torch.no_grad():
        m.reset_memory(2, force=True)
        a, _ = m(x)
        x2 = x.clone()
        x2[:, cut] = (x2[:, cut] + 5) % cfg.model.vocab_size
        m.reset_memory(2, force=True)
        b, _ = m(x2)
    torch.testing.assert_close(a[:, :cut], b[:, :cut], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_set_ablate_changes_nothing_when_read_gate_is_zero_at_init(kind):
    """Fresh models read through a zero-initialised output projection, so ablation is
    a no-op at step 0. That is the intended starting point: any later gap is learned."""
    cfg = tiny_cfg(kind)
    m = MemoryLM(cfg.model).eval()
    x = torch.randint(0, cfg.model.vocab_size, (2, 16))
    with torch.no_grad():
        m.reset_memory(2, force=True)
        on, _ = m(x)
        m.set_ablate(True)
        m.reset_memory(2, force=True)
        off, _ = m(x)
    torch.testing.assert_close(on, off, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_memory_adds_parameters_and_flops(kind):
    base = MemoryLM(tiny_cfg("none").model)
    mem = MemoryLM(tiny_cfg(kind).model)
    assert mem.param_count() > base.param_count()
    assert mem.memory_param_count() > 0
    assert mem.flops_per_token(64) > base.flops_per_token(64)


def test_replace_mlp_drops_the_feedforward():
    cfg = tiny_cfg("product_key", placement="replace_mlp")
    m = MemoryLM(cfg.model)
    assert m.blocks[1].mlp is None and m.blocks[0].mlp is not None


def test_loss_mask_restricts_the_objective():
    cfg = tiny_cfg("none")
    m = MemoryLM(cfg.model)
    x = torch.randint(0, cfg.model.vocab_size, (2, 16))
    y = torch.randint(0, cfg.model.vocab_size, (2, 16))
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, -1] = True
    _, masked = m(x, y, mask)
    _, full = m(x, y)
    assert not torch.isclose(masked, full)
    assert torch.isfinite(masked)


def test_generate_extends_the_sequence():
    cfg = tiny_cfg("slot")
    m = MemoryLM(cfg.model)
    out = m.generate(torch.randint(0, 64, (2, 5)), max_new_tokens=3, top_k=5)
    assert out.shape == (2, 8)


def test_sequence_longer_than_context_is_rejected():
    cfg = tiny_cfg("none")
    m = MemoryLM(cfg.model)
    with pytest.raises(ValueError):
        m(torch.randint(0, 64, (1, cfg.model.max_seq_len + 1)))
