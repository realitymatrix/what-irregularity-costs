# Attributing the A3 / A4 performance gap

Batched timing (see bench/bench_arms.cu) puts the CUDA C++ arm ahead of the
Rust arm on both integrate stages, p50 in ms:

| stage | A3 cuda | A4 rust | ratio |
|---|---|---|---|
| allocate | 0.028 | 0.046 | 1.64x |
| update | 0.249 | 0.303 | 1.22x |

Both arms implement the same algorithm over the same memory and emit the same
PTX instruction family for the CAS and the atomic adds, so the gap is a
code-generation result and worth attributing rather than reporting.

## Refuted: register pressure

The obvious first hypothesis, and wrong. A4 uses **fewer** registers:

| kernel | A3 | A4 |
|---|---|---|
| allocate | 34 | 28 |
| update | 40 | 34 |

Neither arm spills. Lower register use means A4 has at least as much occupancy
headroom, so occupancy cannot explain it being slower.

Reading A4's register counts is awkward: `cuda-oxide` embeds the device image in
the host object rather than emitting a cubin, so `cuobjdump` finds nothing in
the archive. Extract with `ar x`, locate the embedded ELF inside
`*.embed.o` (it starts at a non-zero offset), and dump that.

## Established: A4's update executes 62% more instructions

SASS instruction counts for the update kernel:

    A3 (CUDA C++)   312 instructions
    A4 (Rust)       504 instructions   1.62x

The largest opcode deltas, A4 minus A3:

    LOP3   +27      logic ops
    FFMA   +22
    FSETP  +22      float predicate sets
    BRA    +21      branches
    ISETP   +7
    IMAD   -10

A 1.62x instruction count against a 1.22x runtime is consistent with a kernel
that is partly bound by its atomics rather than by issue rate: extra
instructions cost, but less than proportionally.

## Refuted: `read_volatile` on the hash key

A4 read the hash key with `read_volatile()` where A3 uses a plain load,
reserving `volatile` for the `block_idx` spin. That looked decisive: the probe
loop is the hottest path in the allocate kernel, and an uncached load there
would cost exactly the kind of time the gap represents.

Changing it to a plain `read()` produced **byte-identical SASS**: 528
instructions before and after, with no opcode counts changed at all. LLVM and
NVVM generate the same code either way here, so the marking was never the
cause. The change is kept because a plain read is the correct expression of the
intent, but it buys nothing.

## Refuted: `f32::clamp` NaN semantics

Rust's `clamp` is NaN-propagating and asserts its bounds, where CUDA's
`fminf(fmaxf(...))` is a bare pair of selects, so it looked like a plausible
source of the FSETP delta. Replacing it with a hand-rolled branchless select
changed the instruction count by **zero** (504 to 504); one `FMNMX` became one
`FSEL`. The hypothesis is dead.

## Open: allocate is 1.64x slower on nearly identical instruction counts

    A3 allocate   520 instructions
    A4 allocate   528 instructions   1.02x

Instruction count explains the update gap but explicitly does NOT explain the
allocate gap. Something else dominates there. Candidates not yet tested:

* **The publication spin.** Both arms spin on `block_idx` while a peer
  publishes it, but the loop bodies differ and divergent waiting is sensitive
  to how the compiler schedules the reload.
* **Memory access patterns.** Same addresses, but different coalescing or
  different scheduling of the probe loads.
* **Warp divergence.** A4 has +8 branches in allocate; if they sit inside the
  probe loop the cost is multiplied by probe depth.

### ncu is blocked on this machine

`ncu` is installed but refuses with `ERR_NVGPUCTRPERM`: access to GPU
performance counters is restricted to administrators. Lifting it needs root
either way:

    # /etc/modprobe.d/nvidia-profiling.conf
    options nvidia NVreg_RestrictProfilingToAdminUsers=0
    # then reboot

Until then, warp stall reasons, memory throughput and achieved occupancy are
all unavailable, and the attribution is limited to static analysis of the
generated code.

### REFUTED 2026-08-07: memory-ordering scope

Tested directly and it is not the cause. See docs/CUDA-OXIDE-ATOMIC-LOAD.md for
why the test needed a compiler patch to run at all.

With every load in A4's allocate kernel moved from system scope to GPU scope,
the SASS memory-operation mix becomes:

| kernel | plain LDG.E | LDG.E.SYS | LDG.E(.64).STRONG.SYS | atomics |
|---|---|---|---|---|
| A4 allocate, before | 3 | 0 | **6** | 7 (STRONG.GPU) |
| A4 allocate, after | 3 | 0 | **0** | 7 (STRONG.GPU) |

Runtime, allocate p50 ms, three runs of 20 reps:

| cell | device | before | after |
|---|---|---|---|
| base | 5070 Ti | 0.0460 | 0.045–0.046 |
| base | 5060 | 0.0783 | 0.078 |
| tartan-warm | 5070 Ti | 0.0542 | 0.054–0.055 |
| tartan-warm | 5060 | 0.0746 | 0.073–0.074 |

**Nothing moved.** Confirmed a second time on the full matrix: the identical
sweep (12 cells, 3 passes, both devices, 65,536 scratch slots) re-run with the
scoped-atomic build gives 48 paired A4 comparisons with a mean delta of -0.04%,
a median of +0.01%, and a range of -1.5% to +1.2%.

The control makes it airtight. A3, which was **not** changed between the two
sweeps, moved *more* than A4 did: mean +0.72%, range -2.4% to +41.5%. That
outlier is `r-1.0` on the wide card, one of the large-pool cells already known
to be bimodal. In other words the arm under test moved less than the arm that
could not possibly have moved, so any real effect is well below the noise
floor of the instrument.

The single-cell version of this measurement: Eliminating all six system-scope strongly-ordered loads
changed the ratio from 1.65x to 1.63x on the wide card and 1.45x to 1.44x on
the narrow one, which is noise. Correctness is unaffected: 1160 blocks, mean
surface distance to A3 still 0.000000000 m.

A partial attempt is worth recording because it nearly produced a false
negative. Converting only the `block_idx` spin loads took the count from 6 to 1
and also showed no change — but the one that remained was the probe-loop key
read, which executes every iteration where the spins execute only when a peer
is mid-publication. The SASS made this visible (`IMAD.WIDE.U32 R5, 0x10`
immediately before the surviving `LDG.E.64.STRONG.SYS`, i.e. slot times 16
bytes). Reading the disassembly rather than trusting the source diff is what
caught it.

So the hypothesis is dead, and with it the last candidate on this list. The
allocate gap is unattributed and every mechanism proposed so far has been
tested and eliminated: register pressure, `f32::clamp`, `read_volatile` as
such, instruction count, and now memory scope.

### Superseded: the original scope observation

The SASS memory instructions differ in scope, and the scopes are not cosmetic:
`STRONG.SYS` is a system-scope strongly-ordered access, which is materially
more expensive than a plain or GPU-scope one.

| kernel | plain LDG.E | LDG.E.SYS | LDG.E(.64).STRONG.SYS | atomics |
|---|---|---|---|---|
| A3 allocate | 0 | 6 | 2 | 10 |
| A4 allocate | 3 | 0 | 6 | 7 |

A4 issues **six** system-scope strongly-ordered loads against A3's **two**,
while issuing *fewer* atomics. That is the clearest asymmetry found so far and
it points at how `cuda-oxide` lowers Rust's volatile and atomic accesses versus
how nvcc lowers CUDA's `volatile` qualifier.

It is not proof. Counting instructions says nothing about how often each is
executed or how long each stalls, and the allocate kernel's cost is dominated
by contention that a static count cannot see. Confirming it needs the counters
above, or a variant experiment that removes the remaining `read_volatile` calls
entirely and accepts the correctness loss to isolate their cost.

## The bisect: it is the compare-exchange and the spin

Run 2026-08-07 with `bench/probe_a4_bisect.cu`, at `pts-20k` where the excess is
5.9x and therefore unmissable. Each variant is the real kernel with one
component disabled through a const generic, so the geometry walk and probe loop
are byte-identical and the delta is attributable. Every variant is deliberately
incorrect and exists only to be timed.

Two facts set this up. The excess is a **fixed cost, not a multiplier**: raising
the batch from 8 to 64 amortises A3 (0.005 to 0.003 ms) and leaves A4 pinned at
0.021, which rules out host enqueue, FFI and stream lookup, since all of those
would amortise. And it is **specific to the insert path**: at the same cell the
update kernel is a flat 1.15x with no floor. That shape also explains the
workload dependence. `pts-20k` through `pts-1280k` are all the same R=0.5
sphere, so the block count and therefore the insert count are constant while the
point count varies 64x; a per-insert cost necessarily looks like a fixed offset
across that axis.

| variant | p50 ms | x/A3 | share of excess recovered |
|---|---|---|---|
| A3 CUDA C++ | 0.0035 | 1.00 | — |
| A4 full | 0.0207 | 5.91 | — |
| A4 minus CAS (probe and read only) | 0.0062 | 1.77 | 84% |
| A4 minus publish (CAS and counter, no spin) | 0.0144 | 4.12 | 36% |
| A4 minus spin | 0.0151 | 4.32 | 33% |
| A4 minus fence | 0.0196 | 5.59 | 7% |
| A4 minus counter | 0.0200 | 5.71 | 4% |

Differencing the variants decomposes the 0.0172 ms excess:

| component | cost | share |
|---|---|---|
| the compare-exchange itself | ~0.0075 | **44%** |
| the publication spin | 0.0056 | **33%** |
| the probe path, before any insert | 0.0027 | 16% |
| `threadfence` | ~0.0012 | 7% |
| the shared block counter | ~0.0007 | 4% |
| publishing coordinates and index | ~0.0007 | 4% |

The shares sum to over 100% because the components interact: removing the CAS
also removes every insert, so no thread ever waits on a publication, which is
why `-cas` subsumes most of `-spin`. Treat them as an ordering of suspects, not
as an additive budget.

**Ruled out by this:** the shared counter, the fence, and the coordinate and
index publication. Each is worth under a tenth of the gap, and the fence result
is worth stating twice because A4 emits two `MEMBAR.SC.GPU` against A3's one,
which looked like a lead and is not.

**Not ruled out, and now the whole question:** why A4's `atom.cas.b64` costs
more than A3's, when both emit `ATOMG.E.CAS.64.STRONG.GPU` at the same scope
and both insert exactly 1160 blocks. Two hypotheses remain and this method
cannot separate them:

1. The same number of compare-exchanges, each slower, for example because of
   how they are scheduled against the surrounding loads.
2. More compare-exchanges attempted, for example because the two arms diverge
   differently within a warp and retry more often.

Distinguishing them needs either a device-side counter of CAS attempts added to
both arms, or `ncu`'s warp stall reasons. The counter is the cheaper experiment
and does not need root.

## CORRECTION: the bisect above was run at too small a workload

Giving A3 the same probe-only specialisation, so the two arms' baseline paths
can be compared like with like, changes the conclusion. The `pts-20k` cell used
for the first bisect is dominated by a fixed floor, and the components rank
differently once the workload is large enough for that floor to be irrelevant.

Probe-only, both arms, three sizes:

| points | A3 probe-only | A4 probe-only | ratio |
|---|---|---|---|
| 20,000 | 0.0020 | 0.0062 | 3.10x |
| 320,000 | 0.0146 | 0.0207 | 1.42x |
| 1,280,000 | 0.0573 | 0.0684 | **1.19x** |

The ratio collapses with size, which is the fixed-floor signature again. **A4's
steady-state disadvantage on the geometry walk and probe is 1.19x, not 3.1x.**
The 3.1x was the floor, and quoting it would have been wrong.

The same applies to the headline. A4's allocate is 5.9x A3 at 20k points and
**1.52x at 1.28M**, which matches the sweep's 1.51x for that cell. The 5.9x is a
small-workload artefact and must not be quoted as the gap.

### The ranking inverts: at realistic sizes it is the spin, not the CAS

| component | share of excess at 20k | share at 320k | share at 1.28M |
|---|---|---|---|
| publication spin | 32% | **75%** | **83%** |
| shared counter | 4% | 5% | 5% |
| `threadfence` | 7% | 16% | 13% |

At 20k points the compare-exchange dominated and the spin looked secondary. At
realistic sizes that reverses: the spin is 75% to 83% of the whole gap. The
reason is contention. All these cells are the same R=0.5 sphere with about 1,160
blocks, so raising the point count raises the number of threads racing for the
same blocks, and every thread that loses a race waits for the winner to publish
`block_idx`.

Splitting the remaining excess at 1.28M by path:

| path | A3 | A4 | A4/A3 | share of excess |
|---|---|---|---|---|
| geometry walk and probe | 0.0573 | 0.0684 | 1.19x | 27% |
| insert machinery | 0.0214 | 0.0511 | **2.39x** | 73% |

So roughly a quarter of the gap is a mildly slower baseline path and three
quarters is the insert machinery, almost all of it the spin.

### What this makes actionable

The spin exists to cover a window: a reader can see a published key before the
winner has published the matching `block_idx`, and must wait rather than read
`-1`. Both arms have that window and both pay for it, but A4 pays far more.

Two fixes worth trying, neither of which needs a profiler:

1. **Close the window.** Publish the key and the block index in one 128-bit
   compare-exchange so a reader never observes a half-built entry, which removes
   the spin entirely rather than making it cheaper. The entry is already 16 bytes
   and 16-byte aligned, so this is a layout question, not a redesign.
2. **Do not wait.** Have a loser re-probe from the top instead of spinning on one
   address, converting a serialised wait into more parallel work.

Note that the `recovered` column in the probe output is defined against A3's
FULL kernel, so with A3's probe-only variant present it can exceed 100%. Read
the absolute times, not that column, until it is redefined.

## Instruction counts, static and dynamic: every one favours A4

Measured 2026-08-07 with `bench/probe_cas_count.cu` and `cuobjdump`, at
`pts-20k`. Both arms carry an identical compile-time switch that tallies every
compare-exchange ATTEMPT into `drop_count`, so the dynamic counts are directly
comparable.

| metric | A3 CUDA C++ | A4 Rust | favours |
|---|---|---|---|
| static SASS instructions | 520 | 528 | parity, 1.02x |
| branches (BRA/BSSY/BSYNC/BRX) | 72 | 63 | **A4** |
| global memory ops (LDG/STG) | 11 | 6 | **A4** |
| atomics, static (ATOMG/RED) | 10 | 7 | **A4** |
| registers | 34 | 28 | **A4** |
| stack / local / spills | 0 | 0 | tie |
| **dynamic CAS attempts** | **28,558** | **25,026** | **A4, 12% fewer** |
| blocks built | 1,104 | 1,104 | identical |
| **runtime** | **0.0035 ms** | **0.0207 ms** | **A3, 5.9x** |

There is no instruction-count explanation left. A4 executes fewer
compare-exchanges, from a smaller register budget, with fewer branches and
fewer memory operations, over the same input, producing the same output, and
takes six times as long.

This also kills the second of the two hypotheses the bisect left open. It is
not that A4 attempts more compare-exchanges; it attempts fewer. So the same
instruction, at the same scope, in the same quantity or less, is costing more
each time.

The opcode deltas that remain are real but explain nothing on their own: A4
trades 28 IMAD and 18 FMUL for 23 SEL, 23 LOP3 and 19 FSETP, and moves 27
operands into constant-bank loads that A3 does not use. That is a different
instruction mix for the same arithmetic, not more work.

What is left is the throughput and latency of the atomics themselves: how the
compare-exchange is scheduled, and whether its latency is hidden by other warps.
Nothing measurable without root distinguishes "the CAS issues but stalls longer"
from "the CAS is issued in a more contended pattern", because both present as
the same wall-clock with the same instruction counts. `ncu`'s warp stall
reasons answer it in one run.

One asymmetry to close first, and it is cheap: A4's probe-only variant (0.0062
ms) was compared against A3's *full* kernel (0.0035 ms), because A3 has no
probe-only build. Giving A3 the same `CAS=false` specialisation would make the
probe-path comparison symmetric and would say whether A4's disadvantage starts
before the first compare-exchange or only at it.

## What this does and does not license saying

Established: A4 is slower on both stages under batched timing; it is not a
register-pressure or occupancy effect; the update gap tracks a 1.62x
instruction-count difference; `clamp` is not the cause; memory-ordering scope
is not the cause; and the allocate excess is a fixed per-insert cost that lives
in the compare-exchange (44%) and the publication spin (33%), not in the
counter, the fence or the publication itself.

Not established: why Rust-to-PTX emits 62% more instructions for the same
source algorithm, and why allocate is slower at parity of instruction count.
Neither should be claimed until measured.
