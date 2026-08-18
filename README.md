# mnemos

Large memory models, with the controls attached.

`mnemos` is a small, readable PyTorch library for building and training language
models that carry an explicit memory: a sparse parameter table, a recurrent slot
bank, fast weights that keep learning at test time, or a non-parametric datastore of
past activations. Four memory families, one interface, one training loop.

The second half of the library is the part most memory-model code leaves out. A
memory module adds parameters, adds compute, and adds a pathway, all at once. If you
compare it against the model you started with, you cannot say which of the three
bought the improvement. `mnemos` ships the two controls that separate them and makes
them a first-class command rather than an appendix experiment.

## Install

```bash
cd mnemos && pip install -e ".[dev]"
```

Torch is the only heavy dependency. No tokenizer downloads, no ANN index, no dataset
fetches. Everything in this README runs on a laptop, CPU or Apple MPS.

## Sixty seconds

```bash
python -m mnemos.cli list
python -m mnemos.cli budget --config configs/tiny_pkm.yaml
python -m mnemos.cli train  --config configs/tiny_pkm.yaml --set train.steps=100
python -m mnemos.cli ablate --config configs/tiny_pkm.yaml
```

`budget` tells you what the memory costs before you train anything. `ablate` trains
the memory model, trains a parameter-matched and a FLOP-matched memoryless baseline,
then re-scores the trained memory model with its memory read forced to zero, and
writes one JSON report.

## The memory modules

| kind | family | what it is | state |
|---|---|---|---|
| `product_key` | Lample et al. 2019; Meta memory layers 2024 | `n_keys^2` value slots searched in `O(n_keys)` by factorising keys into halves | parameters only |
| `slot` | RMT / LM2 | a bank of slots carried along the sequence, read by cross-attention, written by gates | per sequence |
| `surprise` | Titans | fast weights updated during the forward pass by the gradient of an associative loss, with momentum and decay | per sequence |
| `knn` | Memorizing Transformers | FIFO datastore of past `(key, value)` activations, exact top-k retrieval | across batches |

All four subclass `MemoryModule` and return a delta on the residual stream:

```python
class MemoryModule(nn.Module):
    def _forward(self, x: Tensor) -> Tensor: ...        # (B, T, D) in, (B, T, D) out
    def reset_memory(self, batch_size, device, dtype): ...
    def flops_per_token(self, seq_len: int) -> float: ...
    def memory_param_count(self) -> int: ...
    def diagnostics(self) -> dict[str, float]: ...
```

Returning a residual delta is what makes the causal control a one-line intervention:
setting `ablate = True` zeroes the delta and changes nothing else about the model.

### Adding your own

```python
from mnemos.registry import register_memory
from mnemos.modules.base import MemoryModule

@register_memory("my_memory")
class MyMemory(MemoryModule):
    def __init__(self, d_model, n_slots=64):
        super().__init__(d_model)
        ...
    def _forward(self, x):
        return ...
```

It is now addressable from YAML as `memory: {kind: my_memory, layers: [2]}`. Nothing
else needs to change. Implement `flops_per_token` and `memory_param_count` if you want
the matched controls to price it correctly.

## Causality

Three of the four modules hold state that is written as the sequence runs, which is
exactly where memory implementations leak the future into the past. Every stateful
module here is chunked so that **a token reads the memory as it stood before its own
chunk began**, and writes land only after the read. Chunk size is a config knob;
`chunk_size: 1` is strict per-token causality at a higher wall-clock cost.

`tests/test_model.py` enforces this directly, for every kind: perturb one token, and
assert that no earlier position's logits move. Any write-before-read bug fails the
suite rather than quietly inflating a benchmark number.

## The controls

**Matched baseline.** No memory, but the feed-forward width is widened by bisection
until the parameter count (or per-token FLOPs) matches the memory model. If the memory
model no longer wins, the win was capacity, not memory.

**Read ablation.** The same trained model, scored with the memory read forced to zero.
This is the causal question a parameter count cannot answer: not "is this model better
than some other model" but "does this model's behaviour depend on the memory at all".
A gap near zero means the pathway is decorative.

The two controls disagree in an informative way, and the disagreement is the reason to
report both. Product-key memory is parameter-heavy and FLOP-light, so a
parameter-matched dense baseline is handed strictly *more* compute than the memory
model, while a FLOP-matched one is handed far fewer parameters. Neither is the honest
single number. There isn't one. `test_param_matching_a_sparse_memory_overpays_in_flops`
pins that asymmetry so it cannot be quietly dropped.

## Tasks

Synthetic probes, generated on the fly, no download:

- `associative_recall` - key/value pairs then queries. The standard memory probe.
- `needle` - one fact buried in filler, retrieved after a long gap.
- `copy` - reproduce a prefix. Bulk retention rather than selection.
- `induction` - emit the token that followed a marker's earlier occurrence.

Plus `text`, a byte-level loader over any `.txt` file you point `data.path` at.

Every task returns `(x, y, mask)` and the loss is scored **only on masked positions**.
This matters more than it looks: on a recall task most positions are free tokens the
model can guess from the marginal, so an unmasked loss dilutes the signal by an order
of magnitude and flatters a memoryless baseline.

## Configs

```yaml
model:
  d_model: 128
  n_layers: 4
  memory:
    kind: product_key
    layers: [2]           # which blocks get a memory
    placement: parallel   # parallel to the MLP, or replace_mlp
    params: {n_keys: 32, topk: 8, half_dim: 32, v_dim: 64}
data:
  kind: associative_recall
  params: {n_pairs: 16, n_queries: 4}
train:
  steps: 400
  lr: 3.0e-3
```

Anything is overridable from the command line: `--set train.steps=50 model.d_model=64`.
Configs are validated on load, so a memory layer index outside the model or a
`seq_len` past the context window fails immediately instead of at step 300.

## Diagnostics

Every memory logs to `runs/<name>/metrics.jsonl` as it trains:

- `read_norm` - the magnitude of what the memory actually returns. If it collapses
  toward zero, the model has learned to route around the memory, and any headline
  number is coming from somewhere else.
- `slot_entropy_frac`, `slots_touched_frac` (product-key) - table utilisation. A large
  table that concentrates on a few slots is not a large memory.
- `write_delta` (slot) - how much each chunk actually changes the bank.
- `surprise`, `mem_norm` (surprise) - associative loss and fast-weight scale.
- `store_size`, `retrieval_hit` (knn) - datastore occupancy and retrieval similarity.

## Reproducing the headline table

```bash
bash scripts/benchmark.sh        # every memory kind + the matched-baseline report
python scripts/report.py         # formats runs/bench/*.json into markdown
```

About an hour on an M-series laptop. `RESULTS.md` holds the output and the reading of
it. Everything in it is a laptop-scale result and is labelled as one.

## Layout

```
mnemos/
  config.py      typed configs, validated on load
  layers.py      RMSNorm, RoPE, causal attention, SwiGLU
  model.py       MemoryLM: decoder-only transformer with memory in named layers
  registry.py    name -> module, so configs can address memories by string
  train.py       Trainer, schedules, checkpoints
  evaluate.py    masked loss/accuracy, read ablation
  ablate.py      matched-baseline construction and budget accounting
  cli.py         train | eval | ablate | budget | list
  modules/       product_key, slot_attention, surprise, knn
  data/          synthetic probes, byte-level text
configs/         one tiny config per memory kind
tests/           94 tests, including the causality suite
```

## A bug this repo found

Early runs on Apple MPS produced negative cross-entropy losses, a mathematical
impossibility, in bursts of a few steps in a hundred. The cause was `non_blocking=True`
on the host-to-device batch transfer: with unpinned source tensors, the copy can read
memory that has already been freed, so a handful of batches arrive as garbage token
ids. Nothing raised. Loss dipped, gradients spiked, and the run continued.

The fix is one word, but the interesting part is that the loop now refuses to continue
past a negative or non-finite loss, with an error that names the likely cause. A
training loop that silently averages over corrupted batches will still produce a
plausible learning curve, and that curve is the thing you would have published.

## Known limits

- The `surprise` module uses a **linear** memory, for which the gradient of the
  associative loss is analytic and the chunked update closes in one matmul. Titans
  uses an MLP memory and backpropagates through it. The MLP variant is a real
  extension, not something this implementation is pretending to already have.
- kNN retrieval is exact and dense. That is the right trade at laptop scale and the
  wrong one past a few hundred thousand entries; swap in an ANN index there.
- The shipped configs are laptop-sized. They demonstrate the machinery. They are not
  evidence about how any of these memories behave at 1B parameters, and the repo does
  not claim otherwise.

## Tests

```bash
python -m pytest -q
```
