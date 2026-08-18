"""Evaluation, plus the diagnostic that most memory papers leave out."""
from __future__ import annotations

import math
from typing import Dict

import torch


@torch.no_grad()
def evaluate(model, dataset, n_batches: int, batch_size: int, generator: torch.Generator,
             device, commit: bool = False) -> Dict[str, float]:
    """Masked loss and masked next-token accuracy over ``n_batches`` fresh batches."""
    was_training = model.training
    model.eval()
    tot_loss, tot_correct, tot_n = 0.0, 0.0, 0.0
    diag_sum: Dict[str, float] = {}
    for _ in range(n_batches):
        x, y, mask = (t.to(device) for t in dataset.batch(batch_size, generator))
        logits, loss = model(x, y, mask)
        n = mask.sum().item()
        tot_loss += loss.item() * n
        tot_correct += ((logits.argmax(-1) == y) & mask).sum().item()
        tot_n += n
        if commit:
            model.commit_memory()
        for k, v in model.memory_diagnostics().items():
            diag_sum[k] = diag_sum.get(k, 0.0) + v
    if was_training:
        model.train()
    tot_n = max(tot_n, 1.0)
    out = {
        "val_loss": tot_loss / tot_n,
        "val_acc": tot_correct / tot_n,
        "val_ppl": math.exp(min(tot_loss / tot_n, 20.0)),
        "scored_tokens": tot_n,
    }
    out.update({k: v / max(n_batches, 1) for k, v in diag_sum.items()})
    return out


@torch.no_grad()
def read_ablation(model, dataset, n_batches: int, batch_size: int, seed: int, device) -> Dict:
    """Score the same model twice, with the memory read on and then forced to zero.

    This is the causal question a parameter count cannot answer: not "does the model
    with memory do better than some other model", but "does *this* model's behaviour
    depend on the memory read at all". A gap near zero means the memory is decorative
    and the gain, if any, came from the extra parameters or depth.
    """
    on = evaluate(model, dataset, n_batches, batch_size,
                  torch.Generator().manual_seed(seed), device)
    model.set_ablate(True)
    off = evaluate(model, dataset, n_batches, batch_size,
                   torch.Generator().manual_seed(seed), device)
    model.set_ablate(False)
    return {
        "memory_on": {k: v for k, v in on.items() if not k.startswith("mem")},
        "memory_off": {k: v for k, v in off.items() if not k.startswith("mem")},
        "delta_loss": off["val_loss"] - on["val_loss"],
        "delta_acc": on["val_acc"] - off["val_acc"],
        "diagnostics": {k: v for k, v in on.items() if k.startswith("mem")},
    }
