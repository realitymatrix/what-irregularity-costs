# The scratch region: hypothesis, test, result

**Confirmed 2026-08-07.** The scratch region explains arm A5's failure to scale
with SM count, and sizing it properly removes a 3.5x cost that had been
attributed to Triton the language.

## Background

`tl.atomic_cas` takes no `mask`, unlike `tl.atomic_add`. A lane that has already
resolved its hash probe therefore cannot be masked off; it still issues a
compare-exchange somewhere. The workaround is to aim those inert CASes at a
scratch address holding a sentinel key that can never satisfy the compare.

The original implementation used `my_scratch = scratch_base + lane`, giving
`BLOCK = 256` scratch addresses. That already fixed the catastrophic case: one
shared slot serialised the whole grid at 10.7 ms against 1.7 ms for the
256-slot region.

But `lane` runs 0..BLOCK, so **every program in the grid used the same 256
addresses**. The number of contenders scales with the grid; the address set does
not.

## The hypothesis

From the two-device sweep (`docs/WORKLOAD-SWEEP.md`), A5's allocate did not
scale with SM count at all (0.89x to 1.07x across a 2.33x SM difference) while
A3 and A4 tracked the machine at 1.8x to 2.1x. The proposed mechanism:

> The total number of atomic operations is fixed by point count and probe bound,
> and they collide on 256 addresses however many SMs issue them. Serialisation
> at those addresses sets the runtime, so more SMs add contenders without adding
> throughput.

This predicts that enlarging the region should both make the kernel faster and
restore device scaling.

## The test

The scratch region size became a **runtime** kernel argument, `scratch_mask`,
with the index changed from `lane` to `(pid * BLOCK + lane) & scratch_mask`.
When `scratch_mask == BLOCK - 1` the `pid * BLOCK` term vanishes modulo BLOCK,
so the historical behaviour is reproduced exactly rather than approximately.
The allocation grew to 1 Mi slots (16 MiB per volume) so every region size is
reachable without a rebuild, and became write-once so `reset()` no longer
copies it.

Base cell, 320k points, allocate p50 ms, n=12:

| scratch slots | 5070 Ti (70 SMs) | 5060 (30 SMs) | narrow/wide |
|---|---|---|---|
| 1 | 10.761 | 9.644 | 0.90x |
| 16 | 10.893 | 9.780 | 0.90x |
| 256 (historical) | 1.812 | 1.611 | 0.89x |
| 4,096 | 0.710 | 1.228 | 1.73x |
| 65,536 | 0.524 | 0.963 | **1.84x** |
| 1,048,576 | 0.535 | 0.976 | 1.82x |
| — A3 CUDA C++ | 0.028 | 0.054 | 1.93x |

## Result: confirmed, on both predictions

**Device scaling is restored.** At 256 slots the narrow card is 0.89x, i.e. very
slightly *faster* despite having 40 fewer SMs. At 65,536 slots it is 1.84x,
which is within noise of A3's 1.93x on the same cell. Triton's allocate scales
with the machine exactly like the CUDA arm once its inert CASes stop colliding.

The inversion at small regions is itself confirmatory: with a serialising
address set, fewer contenders is marginally *better*, which is why the narrow
card won.

**The kernel is 3.5x faster.** 1.812 to 0.524 ms on the 5070 Ti. Saturation is
at roughly 65,536 slots; 1 Mi is no better, and 1 Mi is the first size giving
full per-lane privacy across this grid (1250 programs x 256 lanes = 320,000
addresses), so the benefit plateaus before privacy is complete. Contention is
relieved statistically, not eliminated.

Nothing else moved: block count stays 1160, the analytic gate still passes, and
the update stage is unchanged at 0.352 ms because it uses no scratch.

## What this costs the paper's headline

The published A5/A3 allocate ratio was **73x on the 5070 Ti and 37x on the
5060**. With the region sized properly it is **18.7x and 17.8x**.

Three consequences, and the first is uncomfortable:

1. **A large part of the published gap was an artefact of a poorly-sized
   workaround, not of the language.** The correct claim is that Triton *forces*
   a workaround for the missing `mask`, and that the workaround has a parameter
   whose bad settings cost 20x (1 slot vs 65,536). It is not that Triton is
   inherently 73x here.
2. **The corrected ratio is device-independent**, 18.7x and 17.8x, where the old
   one halved between cards. A ratio that survives a change of machine is a
   better candidate for a language property than one that does not, so the
   corrected number is the stronger result even though it is smaller.
3. **The failure-to-scale finding is now explained rather than merely reported.**
   It was the sweep's most striking result; it turns out to be a property of one
   line of the workaround, and it disappears when that line is fixed.

## What survives, and what it means

The expressiveness finding is unchanged and arguably strengthened, because the
cost is now measured rather than asserted:

> `tl.atomic_cas` takes no `mask`. Resolved lanes cannot be suppressed, so they
> must be given somewhere harmless to write. The programmer must then size and
> index that scratch region, a structure with no counterpart in the CUDA
> implementation, and the cost of getting it wrong spans 20x: 10.8 ms at one
> slot, 1.8 ms at one region per program-block, 0.52 ms once collisions are
> statistically relieved. Getting it wrong also silently destroys the kernel's
> ability to use additional SMs, which no throughput number on a single device
> would reveal.

That last sentence is the finding. A benchmark on one GPU reports 1.8 ms and
looks reasonable. Only the second device shows the kernel had stopped scaling,
and only varying a parameter of the workaround shows why.

## The full sweep, re-run at 65,536 slots

`results/sweep_20260807_084011/`, three passes per device, medians across
passes. Allocate, p50 ms:

| cell | A3 5070Ti | A5 5070Ti | A5/A3 | A3 5060 | A5 5060 | A5/A3 | A5 scaling |
|---|---|---|---|---|---|---|---|
| pts-20k | 0.0052 | 0.0853 | 16.4x | 0.0083 | 0.0949 | 11.4x | 1.11x |
| pts-80k | 0.0106 | 0.1496 | 14.2x | 0.0191 | 0.3094 | 16.2x | 2.07x |
| base | 0.0279 | 0.5277 | 18.9x | 0.0541 | 0.9642 | 17.8x | 1.83x |
| pts-720k | 0.0518 | 1.2218 | 23.6x | 0.1026 | 2.1465 | 20.9x | 1.76x |
| pts-1280k | 0.0799 | 2.0973 | 26.2x | 0.1677 | 3.8446 | 22.9x | 1.83x |
| r-0.25 | 0.0238 | 0.5280 | 22.2x | 0.0443 | 0.9666 | 21.8x | 1.83x |
| r-1.0 | 0.0331 | 0.5363 | 16.2x | 0.0614 | 0.9683 | 15.8x | 1.81x |
| r-2.0 | 0.0606 | 0.5487 | 9.1x | 0.0884 | 0.9831 | 11.1x | 1.80x* |
| lf-sparse | 0.0286 | 0.5313 | 18.5x | 0.0503 | 0.9634 | 19.2x | 1.81x |
| plane-320k | 0.0295 | 0.5351 | 18.1x | 0.0533 | 0.9637 | 18.1x | 1.80x |

\* `r-2.0` runs at batch 4 on the 8 GiB card against 8 on the 16 GiB one, so
its cross-device row is not comparable and the report says so.

**A5 now scales on every cell**, 1.76x to 2.07x, against 1.60x to 2.10x for A3.
The single exception is `pts-20k` at 1.11x, where the kernel is 0.085 ms and
launch overhead dominates for every arm. Before the fix A5 was 0.89x to 1.07x
everywhere.

**The ratio became device-independent.** Per-cell, the two cards now agree
closely: 18.9 vs 17.8 at base, 22.2 vs 21.8 at r-0.25, 18.1 vs 18.1 at
plane-320k. Previously they differed by roughly 2x on every cell. A ratio that
survives a change of machine is a much better candidate for a language property
than one that does not.

**It is still workload-dependent**, 9.1x to 26.2x, and monotonic in the ways
the axes predict: rising with point count (more inert CASes per resolved lane)
and falling with extent (more blocks, so fewer lanes resolve early and the
wasted work is a smaller share). Those are opposite levers and they move the
ratio in opposite directions, which is the shape the mechanism predicts.

The honest headline is therefore a range with the regime named, not a single
number: **Triton's allocate costs 9x to 26x CUDA C++ on this workload family,
about 19x at the baseline cell, on both cards.**

## Follow-up

* The default in `bench_arms` remains `OSN_TRITON_BLOCK - 1` (256), so
  historical numbers remain reproducible. Published tables should come from the
  65,536 run, with the 256-slot row kept as the "obvious implementation"
  comparison.
* `MAX_PROBE` interacts with this: the probe bound sets how many inert CASes
  each resolved lane issues. The `MAX_PROBE = 32` measurements were taken at 256
  slots and are therefore also pessimistic.
* Whether A4's unattributed allocate gap moves at 65,536 slots is untested; it
  should not, since A4 uses no scratch, and confirming that is a cheap control.
