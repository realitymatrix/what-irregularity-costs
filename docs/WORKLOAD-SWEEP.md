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

### Status

* **Established:** the loss is real, monotonic in load factor, and absent at the
  load factor all prior numbers were taken at.
* **Established:** probe exhaustion has no drop accounting, unlike pool
  exhaustion.
* **Pending confirmation:** a `MAX_PROBE = 32` rebuild should reduce or remove
  the loss at load factor 0.283. Compiling.
* **Not yet done:** the missing `drop_count` increment on probe exhaustion.
  Once added, the cell becomes reportable-with-caveat instead of invalid, and
  the loss becomes a measured quantity rather than an inferred one.

Until that is fixed, `loadfactor` cells above roughly 0.05 are marked INVALID
and excluded from the CSV, which is the correct behaviour: an arm that builds
fewer blocks is doing less work, and timing it against arms that did all the
work would report the bug as a speedup.

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
