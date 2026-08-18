# Design notes

Three decisions shape everything else in this repo.

## 1. A memory is a residual read

Every memory module takes the hidden states of a layer and returns a delta of the same
shape, added into the residual stream. It is not a new block type, not a replacement
attention, not a wrapper around the model.

The payoff is the control. `ablate = True` returns zeros, and the model is otherwise
bit-identical: same weights, same shapes, same everything. So the difference between
the two scores is attributable to the memory read and to nothing else. If the memory
were structural instead - a different block, a different attention pattern - there
would be no way to switch it off without changing the model you are measuring.

The cost is that a memory cannot change the model's shape. A memory that needs to
extend the sequence, or run its own attention over the full context, does not fit this
interface. That is a deliberate boundary, not an oversight.

## 2. Chunked state, read before write

The stateful modules process the sequence in chunks. Within a chunk, every token reads
the state as it stood *before the chunk began*; the write derived from the chunk lands
afterwards. So a token can never read a write derived from itself or from anything
later.

This is conservative. With `chunk_size = 32`, a token at position 40 cannot see
position 33, even though position 33 is legitimately in its past. Strict per-token
causality is available (`chunk_size: 1`) and costs a Python-level loop over the
sequence. The default trades a little recall for a lot of wall-clock, and never trades
in the direction that inflates a result.

Write-before-read is the single most common way a memory implementation quietly cheats.
It is tested for, per module, in `tests/test_model.py::test_no_future_leakage` and
`test_mid_sequence_perturbation_is_causal`.

## 3. Read gates start closed

Every memory module's output projection is zero-initialised, so a freshly built model
is exactly its memoryless counterpart and the memory read has to earn its way in
through training. Two consequences worth knowing:

- Early training is stable. The memory cannot inject noise into a model that has not
  learned to use it yet.
- `read_norm` at the end of training is a real measurement. It started at zero; whatever
  it is now, the optimiser put it there. A `read_norm` that stays near zero is the model
  telling you it found the memory useless.

## What the matched baselines cannot do

The bisection in `ablate.py` matches a budget by widening `d_ff`. It matches parameters
*or* FLOPs, never both, because for a sparse memory those are different models. It also
does not match:

- **Optimisation difficulty.** A wider MLP and a memory table do not have the same loss
  landscape, and both were tuned at the same learning rate. Sweep per arm before
  reporting a margin.
- **Depth.** Widening keeps `n_layers` fixed. If your memory buys effective depth, a
  width-matched baseline understates the control.

Both are documented rather than papered over. A matched baseline is a much better
control than none, and it is not a proof.
