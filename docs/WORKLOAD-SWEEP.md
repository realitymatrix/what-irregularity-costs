# The workload sweep

Added 2026-07-30. Until now every published number came from one scene, a
sphere at 320k points. That is not enough to call a ratio a property of a
*language*: it could equally be a property of that scene's contention pattern.
This document describes the matrix, why each axis exists, and what the first
run found.

Run it with:

    tools/sweep.sh build                  # all cells, all devices
    tools/sweep.sh build --axis loadfactor
    tools/sweep_report.py results/sweep_*/sweep.csv

## The axes

Each cell exists to answer a stated question. The matrix is an explicit list
rather than a cross product, because a full cross product of four axes is
mostly uninformative cells whose main effect is to make the run long enough
that nobody repeats it.

| axis | cells | what it varies | what it tests |
|---|---|---|---|
| baseline | `base` | nothing | keeps every previously published number comparable |
| points | `pts-20k` … `pts-1280k` | point count at fixed geometry | contention rises, hash work constant |
| extent | `r-0.25`, `r-1.0`, `r-2.0` | sphere radius at fixed angular resolution | hash pressure rises, per-voxel contention falls |
| loadfactor | `lf-hi` … `lf-sparse` | pool size, hence table occupancy | probe depth |
| shape | `plane-320k` | sphere vs plane at equal point count | block contiguity and hash clustering |

`points` and `extent` are deliberate opposites. One raises contention while
holding the hash workload fixed; the other raises hash pressure while lowering
contention. An arm's behaviour across both separates the accumulation cost from
the hash cost, which no single axis can do.

The `loadfactor` axis is the sharpest test of the published Triton result. The
prediction was: Triton has no per-lane early exit, so it pays the full probe
bound regardless of load, while CUDA C++ exits as soon as a lane resolves and
therefore pays actual probe depth, which rises with load. The gap should
therefore **narrow** as load factor rises.

Load-factor cells size the pool from a measured block count rather than a
guess, so the achieved load factor is reported rather than assumed.

## Device axis

One process per device, driven by `tools/sweep.sh`. Both the Triton cubin and
the cuda-oxide module bind to the CUDA context current at load time, so
switching devices inside a process would need a module reload per arm per
device. A process per device removes the failure mode.

Devices run sequentially, never concurrently: the harness refuses to measure a
shared device, and two sweeps at once would contend for host CPU because the
driver is set to spin-wait.

**The two local cards are the same architecture.** RTX 5070 Ti (70 SMs) and
RTX 5060 (30 SMs) are both sm_120, so all three toolchains emit identical
machine code for both. This axis varies machine width and bandwidth at fixed
codegen. It is **not** the second-architecture pass and must not be written up
as one; that still needs a different generation, an L4 (sm_89) being the
cheapest.

It is still worth running, because the arms are bound by different things: an
issue-bound arm should track the 2.33x SM ratio, and a contention-bound arm
should not. `sweep_report.py --stage allocate` prints that comparison directly.

## First finding: Triton silently loses blocks at realistic load factors

The harness's validity gate fired on its first run, on a divergence that the
single-scene benchmark could never have exposed.

| load factor | A3 blocks | A5 blocks | missing |
|---|---|---|---|
| 0.018 | 1160 | 1160 | 0 |
| 0.035 (old baseline) | 1160 | 1160 | 0 |
| 0.071 | 1160 | 1159 | 1 |
| 0.142 | 1160 | 1159 | 1 |
| 0.283 | 1160 | 1147 | **13** |

Monotonic in load factor, and zero at the load factor everything was previously
measured at. The old baseline sits at 0.035, which is why this never appeared.

### Mechanism

The two arms bound their probe loops differently, and they had to:

    A3 (CUDA C++)   for (uint32_t probe = 0; probe < size; ++probe)
    A5 (Triton)     for p in tl.static_range(0, MAX_PROBE)     # MAX_PROBE = 8

A3 probes the entire table, so it always finds a slot when one exists, and
increments `drop_count` when the pool itself is exhausted. A5 probes eight
slots and then stops. On exhaustion it does not report: the pool-exhaustion
path at `tsdf_kernels.py:218` increments `drop_count`, but there is no
equivalent for probe exhaustion, so the block is lost with no signal.

The missing accounting is a plain bug and should be fixed. **The bound is not.**
`tl.static_range` is a statically unrolled loop, so the trip count must be a
compile-time constant; a data-dependent probe over the whole table is not
expressible. Making the bound equal to the table size would mean unrolling
thousands of iterations, which is not viable — a rebuild at merely
`MAX_PROBE = 32` did not finish compiling in ten minutes, against seconds at 8.

### Why this matters more than the performance number

This upgrades the Triton finding from a cost to a correctness constraint. The
published version says Triton's missing per-lane early exit costs 8.38x on a
probe loop. The stronger version is:

> Triton cannot express an unbounded probe at all. The programmer must pick a
> fixed bound. A bound small enough to compile is small enough to lose data at
> load factors CUDA C++ handles exactly, and the loss is silent and
> load-dependent, so it will not appear in a sparse benchmark.

That is a considerably more useful thing to tell a practitioner than a
throughput ratio, and it was found by varying a workload parameter rather than
by reading the language reference.

### Confirmed by rebuild

A `MAX_PROBE = 32` rebuild removes the loss completely: all four load-factor
cells become valid, including 0.283 where eight probes lost 13 blocks. The
mechanism is therefore established, not inferred.

It also prices the fix. A5 allocate, same cells, p50 ms:

| MAX_PROBE | compile | cubin | LF 0.283 time | blocks lost |
|---|---|---|---|---|
| 8 | seconds | 683 KB | 2.04 | 13 |
| 32 | ~20 min | 2.51 MB | 9.32 | 0 |

Roughly 4.6x the time for 4x the bound, which confirms the published
linear-in-`MAX_PROBE` result at a second point. Correctness at this load factor
costs Triton a further 4.6x on top of the gap it already had, and the cubin
grows 3.7x because the loop is statically unrolled.

### Status

* **Established:** the loss is real, monotonic in load factor, absent at the
  load factor all prior numbers were taken at, and removed by raising the bound.
* **Established:** probe exhaustion has no drop accounting, unlike pool
  exhaustion.
* **Fixed:** A5 now increments `drop_count` on probe exhaustion, matching how
  the CUDA arm reports its own equivalent failure. Costs 2% cubin size and no
  registers. The correctness gate still passes at 0.000015 m mean radius error
  and 0.000000019 m mean surface distance to A3.

The counter turns out to be far more sensitive than the block-count comparison,
because a drop is one lost point-voxel **contribution** rather than one lost
block, and many points probe toward the same block:

| load factor | drops reported | blocks missing |
|---|---|---|
| 0.018 | 0 | 0 |
| 0.071 | 155 | 1 |
| 0.142 | 155 | 1 |
| 0.283 | 26,940 | 13 |

`loadfactor` cells above roughly 0.05 are still marked INVALID at
`MAX_PROBE = 8` and excluded from the CSV, which remains correct: an arm that
builds fewer blocks is doing less work, and timing it against arms that did all
the work would report the bug as a speedup. The difference is that the loss is
now a measured quantity rather than a silent divergence.

## Second finding: the load-factor prediction FAILED

The `loadfactor` axis was built to test a specific prediction, stated above and
in `PAPER-FRAMING.md`: because Triton pays a fixed probe bound while CUDA C++
exits early and therefore pays actual probe depth, the A5/A3 gap should
**narrow** as load factor rises.

It does not. At `MAX_PROBE = 32`, allocate p50 in ms, device 0, n=15:

| load factor | A3 cuda | A4 rust | A5 triton | A4/A3 | A5/A3 |
|---|---|---|---|---|---|
| 0.018 | 0.028 | 0.053 | 7.631 | 1.89x | 270.6x |
| 0.071 | 0.029 | 0.046 | 8.566 | 1.60x | 300.1x |
| 0.142 | 0.030 | 0.050 | 7.661 | 1.69x | 256.5x |
| 0.283 | 0.032 | 0.066 | 9.322 | 2.03x | 288.5x |

The A5/A3 ratio is flat and noisy across a 16x change in load factor. No trend.

### Why: the lever was too weak, and structurally so

A3's own allocate barely moves, 0.028 to 0.032 ms, 14% across the whole range.
For linear probing the expected probe count is about 1/(1-a), so a rises from
1.02 probes at load factor 0.018 to 1.39 at 0.283. CUDA's early exit was never
saving much here, because there was almost nothing to exit early from.

This is not a matter of adding more cells. **The design caps the reachable load
factor.** The table is sized `next_pow2(pool * 2)`, so with a pool that must
hold every block, load factor cannot exceed 0.5, and power-of-two rounding
usually drops it well below: 1160 blocks need a pool of at least 1160, giving a
4096-slot table and 0.283.

Testing the prediction properly needs load factors around 0.7 to 0.9, where
probe chains actually blow up, and that needs the table size decoupled from the
pool size. That is a change to the volume, not to the sweep, and it is a real
prerequisite rather than a nice-to-have: **the published explanation of the
Triton allocate gap has not been tested in the regime where it makes its
sharpest prediction.**

### What this changes in the write-up

Nothing about the mechanism is refuted. The 8.38x linear-in-`MAX_PROBE` result
stands and is now confirmed at a second bound. What is refuted is a *corollary*
predicted from it, and the honest position is:

> Triton's fixed probe bound costs linearly in the bound, confirmed at two
> bounds. The prediction that this cost would shrink relative to CUDA C++ at
> higher table occupancy was tested and not observed, because the achievable
> occupancy range in this design is too narrow for CUDA's early exit to matter.
> The prediction remains untested rather than refuted.

## Fourth finding: Triton's allocate does not scale with SM count. At all.

The device axis, run properly (see the measurement bug below), is the sharpest
result in the sweep. Allocate, p50 ms, narrow = RTX 5060 (30 SMs), wide =
RTX 5070 Ti (70 SMs), SM ratio 2.33x:

| cell | A3 cuda | A4 rust | A5 triton |
|---|---|---|---|
| pts-20k | 1.60x | 1.02x | 1.03x |
| pts-80k | 1.80x | 1.35x | 1.07x |
| base | 1.91x | 1.68x | 0.99x |
| pts-720k | 1.93x | 1.72x | 0.98x |
| pts-1280k | 2.08x | 1.80x | 0.98x |
| r-0.25 | 1.84x | 1.49x | 0.98x |
| r-1.0 | 1.85x | 1.64x | 0.89x |
| r-2.0 | 1.45x | 1.64x | 0.89x |
| plane-320k | 1.85x | 1.67x | 0.99x |

A3 and A4 track the machine, reaching 1.8x to 2.1x of a possible 2.33x. **A5 is
flat, and on three cells is slightly faster on the smaller card.** Adding 40 SMs
buys Triton's allocate nothing.

The contrast with the *update* stage is what makes this interpretable. There all
three arms scale together, 1.5x to 1.8x, A5 included. So this is not "Triton
does not scale"; it is specific to the irregular kernel.

### This inverts the prediction made before the run

The stated expectation was the reverse: that Triton's allocate is issue-bound,
because it executes `MAX_PROBE` iterations regardless of data, and would
therefore track SM count; and that CUDA's allocate is contention-bound, because
it exits early and leaves only atomics on hot slots, and would therefore not.

Measured, both halves are wrong. CUDA scales nearly linearly with the machine;
Triton does not scale at all.

### Mechanism: CONFIRMED, see docs/SCRATCH-HYPOTHESIS.md

The candidate below was tested and confirmed. Enlarging the scratch region from
256 slots to 65,536 restores device scaling to 1.84x, against A3's 1.93x on the
same cell, and makes the kernel 3.5x faster (1.812 to 0.524 ms). The
failure-to-scale reported here is a property of the workaround's sizing, not of
Triton's allocate as such, and the A5/A3 ratios in this document are therefore
pessimistic by roughly 3.5x. Original text follows.

The leading candidate is the scratch region. Because `tl.atomic_cas` takes no
mask, every resolved lane still issues a CAS, aimed at one of
`kScratchSlots = 256` addresses. The total number of atomic operations is fixed
by point count and probe bound, and they collide on 256 addresses regardless of
how many SMs issue them. Serialisation at those addresses would then set the
runtime, and more SMs would add contenders without adding throughput, which is
exactly the observed shape.

This predicts that varying `kScratchSlots` moves the device-scaling behaviour,
and that a large enough region restores scaling. **Both predictions held.**

### It also changes how the headline ratio must be quoted

A5/A3 on allocate is about 72x on the 5070 Ti and about 37x on the 5060. The
gap halves on the smaller card, **but not because Triton improved**: Triton is
unchanged and CUDA C++ got slower. Any quoted ratio must name the device, and
the write-up should quote the wide card, because that is where the language
choice costs the most and because it is the less favourable number for the
argument being made.

## Fifth finding, methodological: the device axis was silently fake

The first full sweep reported both devices and produced a clean null result:
every arm at 1.00x speedup across a 2.33x SM difference. It was wrong. Device 1
was idle throughout, at 9.3 W falling to 8.3 W with clocks at 180 MHz, while
the harness printed its name and SM count in the header.

Arm A4's loader calls `CudaContext::new(0)`
(`crates/tsdf-rust-cuda/src/lib.rs:497`), which hardcodes device 0 and, being a
driver-API context creation, becomes current for the entire process. `a4_init()`
runs before the first cell, so every arm afterwards ran on device 0 while
`cudaSetDevice(1)` sat ignored.

What makes this worth recording is that the wrong answer was *publishable*.
"Identical performance across a 2.33x SM difference" is a striking result with a
ready-made explanation: latency-bound, contention-bound, pick one. Nothing in
the timings looked anomalous. It was caught only because the null result was
surprising enough to check the power draw, and the correct instinct there is
that a surprising result is a reason to verify the instrument first.

Two fixes, both kept:

* `tools/sweep.sh` selects the device with `CUDA_VISIBLE_DEVICES` rather than
  `--device`, so the process sees one GPU and a hardcoded ordinal 0 is correct
  by construction, whatever any dependency does.
* `bench_arms` calls `cuCtxGetDevice` after loading the arms and refuses to run
  if the context is not on the requested device. `cudaGetDevice` would not have
  caught this: it reports the runtime's intent, not the context in force.

## Third finding: the A4 allocate ratio is workload-dependent

A4/A3 on allocate ranges 1.03x to 4.07x across the full sweep on the 5070 Ti,
and 1.32x to 2.60x on the 5060. The single previously published figure was
1.64x. The extremes are the small-scene cells, where 20k points leave the
kernel too short to amortise anything, and `r-2.0`, where the surface spreads
over enough blocks that both arms are memory-bound and converge to 1.03x.

That number should therefore be reported as a range with the workload and the
device stated, not as a constant. It also raises the bar on attributing it: an unattributed
gap that is also workload-dependent cannot be explained by a static property of
the generated code alone, which weakens the memory-ordering-scope hypothesis
somewhat, since instruction scope does not vary with load factor.

## Validity gates

A cell is discarded, loudly, if:

* any arm's `drop_count` is non-zero, meaning the pool was too small; or
* any arm's block count disagrees with A3's.

Block counts are sampled every repetition rather than once, because an arm may
drop blocks only under contention, which is precisely the regime the sweep
exists to explore.

`BATCH` is reduced automatically when the cell's volumes will not fit in device
memory, and the reduction is printed. A cell measured at a different batch size
is not directly comparable to one that was not, so this is reported rather than
silently applied.
