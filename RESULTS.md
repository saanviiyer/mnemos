# Results

Laptop-scale numbers from `scripts/benchmark.sh`. Regenerate with `python scripts/report.py`.

**Read this as a demonstration of the machinery, not as evidence about how these memory
families behave at scale.** Every model here is roughly 0.8M parameters trained for
3000 steps on one synthetic task with one learning rate. That is enough to show the
controls working. It is not enough to rank the five memory designs against each other,
and a ranking is not claimed.

## Setup

- Task: `associative_recall`, 12 key/value pairs, 4 queries, sequence length 36
- Model: 4 layers, `d_model` 128, `d_ff` 256, byte-ish vocabulary of 64
- Memory sits in block 2, parallel to the feed-forward branch
- 3000 steps, batch 128, AdamW, cosine schedule, one seed
- Loss and accuracy are scored **only** on the 4 answer positions per sequence

Chance on this task is a loss of `ln(31) = 3.43` and an accuracy of `1/31 = 0.032`.

## Five arms on the same task

| memory | params | FLOPs/token | val loss | val acc | read norm |
|---|---|---|---|---|---|
| `none (control)` | 665k | 1.36M | 2.284 | 0.167 | n/a |
| `product_key` | 804k | 1.51M | 2.277 | 0.175 | 5.161 |
| `slot` | 817k | 1.43M | 2.282 | 0.167 | 0.074 |
| `surprise` | 698k | 1.46M | 2.283 | 0.171 | 0.885 |
| `knn` | 714k | 2.46M | 2.282 | 0.171 | 0.220 |

All five arms land inside 0.007 nats and 0.9 points of accuracy. With one seed that is
not a ranking and it is not presented as one. They are also not converged: every arm
descended monotonically through all four evals and none flattened.

```
none       2.414/0.121  2.348/0.136  2.304/0.156  2.284/0.167
pkm        2.418/0.130  2.333/0.151  2.297/0.165  2.277/0.175
slot       2.422/0.117  2.343/0.130  2.302/0.159  2.282/0.167
surprise   2.423/0.127  2.339/0.147  2.297/0.160  2.283/0.171
knn        2.417/0.123  2.345/0.141  2.302/0.162  2.282/0.171
```

So the reading is "all five undertrained at this budget", not "memory does not help".

(The `knn` arm was killed at step 2400 by a session teardown and restarted from scratch.
Its first attempt's metrics used to be appended into the same file, which stitched a
saw-tooth curve out of two runs; the trainer now rotates a previous attempt aside to
`metrics.jsonl.1` instead. The curve above is the surviving run only.)

## The matched controls

Product-key memory costs 1.210x the parameters and 1.110x the per-token FLOPs of the
memoryless model. Widening `d_ff` to match each budget separately:

| arm | d_ff | params | FLOPs/token | val loss | val acc |
|---|---|---|---|---|---|
| product_key | - | 804k | 1.51M | 2.277 | 0.172 |
| matched on params | 347 | 804k | 1.64M | 2.280 | 0.170 |
| matched on FLOPs | 305 | 740k | 1.51M | 2.284 | 0.164 |

- Matched on params: `d_ff` 256 -> 347, parameter ratio 1.0005, **FLOP ratio 1.086**
- Matched on FLOPs: `d_ff` 256 -> 305, FLOP ratio 1.0007, **parameter ratio 0.920**

Neither control is neutral, as `ablate.py` documents. The parameter-matched
baseline is handed 8.6% more compute than the memory model; the FLOP-matched one is
handed 8% fewer parameters. Product-key memory beats the parameter-matched baseline by
0.003 nats and 0.2 points, and the FLOP-matched one by 0.006 nats and 0.7 points. Both
margins are smaller than the spread between arms in the table above, which is to say
both are noise at this budget.

## The read ablation is the part worth keeping

Two independent trainings of the same product-key config, memory read on then forced to
zero:

| run | scored tokens | val loss on -> off | val acc on -> off | read norm |
|---|---|---|---|---|
| standalone | 16384 | 2.2736 -> 2.2754 (+0.0018) | 0.17505 -> 0.17487 | 5.16 |
| inside `ablate` | 8192 | 2.2772 -> 2.2774 (+0.0002) | 0.17163 -> 0.17163 | 0.47 |

Meanwhile every utilisation diagnostic says the memory is working hard: 82% and 96% of
the 1024 slots touched, slot entropy 0.83 and 0.90 of maximum. The table is addressed
broadly, the read carries real magnitude, and deleting it entirely costs the model
essentially nothing.

That is the gap this repo exists to expose. Utilisation metrics measure whether a memory
is *addressed*. They cannot measure whether the model *depends* on it. Here they
disagree completely and only the causal test noticed.

## A measurement caveat found by running the same config twice

The two product-key runs above share a config and a seed. Their validation losses agree
to four decimals (2.27706 vs 2.27721). Their `read_norm` differs by 11x (5.16 vs 0.47).

On MPS, no seed reproduces bit-for-bit across executions, and `read_norm` turns out to be
one of the quantities that does not survive that. So **`read_norm` magnitude is not a
reportable single-run statistic** in this repo, and any claim resting on it needs several
executions. What did survive both runs is the thing being claimed: the ablation delta was
~0 either way.


## Caveats that would have to be closed before any of this is a claim

- **One seed.** No error bars. Differences smaller than seed noise are not differences.
- **One learning rate across all arms.** A shared learning rate silently handicaps
  whichever architecture wanted a different one. Sweep per arm before quoting a margin.
- **One task, one sequence length.** Associative recall is where an explicit memory
  should look best. That is why it is the demo and why it is not the evidence.
- **One scale.** The interesting question about large memory models is what happens as
  the table grows relative to the dense parameters, and none of these runs vary that.
