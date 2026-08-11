# Triton's allocate cost does not respond to the work

Found while chasing an apparent 85x on a newly added environment. The 85x was
an artefact. What it led to is not, and it is a sharper statement of this
paper's thesis than anything in v1.

## The 85x was the scratch region

A one-off run of `diner-p000` showed Triton at 85x on allocate, well outside the
11-32x the sweep reports. It was measured before the corrected scratch size
became the default, so it is the historical 256-slot configuration:

| cell | scratch 256 | scratch 65536 |
|---|---|---|
| base (sphere, 320k) | 60.5x | 18.0x |
| diner-p000 (real, 394k) | **84.9x** | 26.0x |

Nothing to do with the environment. Closed.

## The real question it exposed

At the correct setting `diner-p000` is still 26x where the sphere is 18x, and
the sweep's nine real cells sit at 17-32x against 11-32x overall. The paper
reports that as real depth being at the worse end of Triton's range. That
phrasing is true and the explanation implied by it is wrong.

## What is actually happening

Six cells in the sweep hold the point count at 320k and vary only how much
insertion work those points create:

| cell | blocks | CUDA C++ | Triton |
|---|---|---|---|
| r-0.25 | 280 | 0.023 ms | 0.501 ms |
| plane-320k | 1,060 | 0.029 | 0.501 |
| base | 1,160 | 0.028 | 0.501 |
| lf-sparse | 1,160 | 0.029 | 0.505 |
| r-1.0 | 4,752 | 0.032 | 0.503 |
| r-2.0 | 18,160 | 0.045 | 0.507 |

Blocks vary by **65x**. CUDA C++ varies by 1.90x, because more blocks means more
insertions and insertions are work. **Triton varies by 1.013x.**

A least-squares fit of `allocate_ms = a + b * points + c * blocks` over all 19
cells says the same thing. Triton costs 1.49 ms per million points and its block
coefficient is negligible: at a typical cell, block count accounts for 2.4% of
its predicted time against 97.6% from point count alone.

**Triton's irregular-stage cost is a function of how many threads are launched,
not of what those threads accomplish.** Every lane runs the probe loop to its
compile-time bound whether it inserts a block, finds one already there, or has
nothing to do. The bound is paid unconditionally, so the kernel cannot get
faster when the problem gets easier.

## Which is why real data looks worse

The real cells are warm: they integrate a frame into a volume that already holds
eight frames, so most of their probes hit rather than insert. Cold twins of four
of them, same frames into an empty volume, separate the two:

| cell | CUDA cold -> warm | Triton cold -> warm | ratio cold -> warm |
|---|---|---|---|
| tartan-p001 | 0.031 -> 0.019 ms | 0.637 -> 0.635 ms | 20.5x -> 33.4x |
| tartan-p003 | 0.039 -> 0.023 | 0.660 -> 0.658 | 16.9x -> 28.6x |
| diner-p000 | 0.034 -> 0.025 | 0.651 -> 0.650 | 19.1x -> 26.0x |
| diner-p003 | 0.040 -> 0.023 | 0.663 -> 0.664 | 16.6x -> 28.9x |

Warming the volume makes CUDA C++ 1.4-1.7x **faster**, because a hit is cheaper
than an insertion. Triton does not move at all: four pairs, largest change
0.3%. The ratio worsens entirely through the denominator.

So the correct statement is not that real depth is harder for Triton. It is
that a warm volume is easier for everything that can exploit it, and Triton
cannot. `tartan-cold` was the clue sitting in the published sweep the whole
time: the one real cell without a warm pre-fill, and the one real cell that
lands at 17.8x among the synthetic ones rather than up with its neighbours.

## Why this matters more than a ratio

A ratio measured on one workload invites the reader to assume it transfers. This
does not transfer, and it does not transfer in a direction that flatters the
measurement: the gap widens exactly as the workload becomes more like a running
pipeline, where the volume is warm for every frame after the first and most
probes hit.

It also makes the expressiveness argument quantitative. The paper says Triton
has no per-lane early exit. The cost of that is now measurable as a floor: 1.49
ms per million points that no amount of reduced work removes.

## Reproducing

The cold twins are cells on a `coldwarm` axis:

```bash
build/bench_arms --device 0 --reps 5 \
  --cells diner-p000,diner-p000-cold,tartan-p003,tartan-p003-cold
```

The 320k comparison needs no new run; it is six cells of the published sweep in
`results/sweep_20260807_202302/`.

## Status

Not in arXiv:2608.08287v1. This is the strongest candidate for a v2, alongside a
third environment.
