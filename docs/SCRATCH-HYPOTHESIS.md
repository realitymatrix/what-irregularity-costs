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

## Follow-up

* The default in `bench_arms` remains `OSN_TRITON_BLOCK - 1` (256), so
  historical numbers remain reproducible. The sweep should be re-run at 65,536
  and the paper's tables regenerated from that, with the 256-slot row kept as
  the "obvious implementation" comparison.
* `MAX_PROBE` interacts with this: the probe bound sets how many inert CASes
  each resolved lane issues. The `MAX_PROBE = 32` measurements were taken at 256
  slots and are therefore also pessimistic.
* Whether A4's unattributed allocate gap moves at 65,536 slots is untested; it
  should not, since A4 uses no scratch, and confirming that is a cheap control.
