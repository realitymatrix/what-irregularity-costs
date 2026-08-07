# Paper framing

Reframed 2026-07-30 after the depth measurements killed the crossover claim.
Rewritten 2026-08-07 after the last unattributed gap was attributed. Earlier
revisions of this file carry numbers that later measurements corrected; where a
figure changed, the correction and its cause are stated rather than the old
value quietly replaced.

## The claim

**What does it cost to express an irregular, atomics-heavy GPU workload in CUDA
C++, in Rust, and in Triton?**

A hash-blocked TSDF integrate path is a good vehicle because it is the opposite
of the tiled dense linear algebra these tools are usually compared on: an
open-addressed hash table with compare-exchange insertion, per-lane variable
probe depth, scattered atomic accumulation, and no regular tile structure
anywhere.

Answered on two axes, both load-bearing. **Performance**, measured and
attributed to a mechanism. **Expressiveness**, what each language cannot say.
They are not separate: every performance result in this paper is an
expressiveness cost made numerical, which is what lets the numbers be explained
rather than merely reported.

## Why the original framing died

The project was scoped around a crossover: with a heavyweight depth model the
fusion backend is irrelevant, but with a real-time model fusion becomes the
bottleneck. Measured across an 11.8x depth range, fusion is 0.18% to 2.0% of the
pipeline and never approaches being the bottleneck. No caveat rescues it: the
fastest available depth model, at fewer pixels than the dataset's native
resolution, still costs 48x the fastest fusion arm.

That leaves the language comparison, which was always the more novel half, and
promotes it from an input to a budget argument into the result itself. Which is
also what forced the attribution work: a ratio feeding a crossover could be
reported bare; a ratio that IS the paper cannot.

## 1. Performance

Integrate path only; extraction is excluded and the reason is in
docs/ARM-SCOPE.md. Batched, interleaved, exclusive device, medians of three
passes over a 12-cell sweep on two GPUs (docs/WORKLOAD-SWEEP.md). Final state,
after every fix described below.

**Full integrate path (allocate + update), p50 ms:**

| workload | device | A3 CUDA C++ | A4 Rust | A5 Triton |
|---|---|---|---|---|
| TartanAir warm | 5070 Ti | 0.3878 | 0.3960 (**1.02x**) | 1.1333 (2.92x) |
| TartanAir warm | 5060 | 0.7630 | 0.7680 (**1.01x**) | 2.2286 (2.92x) |
| TartanAir cold | 5070 Ti | 0.4726 | 0.4863 (**1.03x**) | 1.2147 (2.57x) |
| TartanAir cold | 5060 | 0.9041 | 0.9154 (**1.01x**) | 2.3342 (2.58x) |
| synthetic sphere | 5070 Ti | 0.2781 | 0.3077 (1.11x) | 0.8282 (2.98x) |
| synthetic sphere | 5060 | 0.5107 | 0.5383 (1.05x) | 1.5258 (2.99x) |

**By stage**, which is where the story is:

| stage | A4 / A3 | A5 / A3 |
|---|---|---|
| allocate (irregular: probe, CAS, publication) | 1.12–1.33x | **17.3–28.6x** |
| update (regular: walk and accumulate) | 1.00–1.08x | 1.23–1.41x |

The shape is the finding. On the regular kernel all three languages land within
1.4x of each other. On the irregular one Rust stays within a third and Triton is
an order of magnitude out. **Choosing a GPU language costs nothing on the tiled
work everyone benchmarks, and a great deal on the work nobody does.**

Three attributions follow, and one honest residual.

### Triton's allocate gap: 17–29x, attributed

Two mechanisms, both consequences of things the language cannot express, both
confirmed, and they compound.

*No per-lane early exit.* A probe loop must run to a fixed bound with a `done`
mask, so every lane pays worst-case probe depth. Cost is linear in the bound:
8.38x for an 8x bound, confirmed again at `MAX_PROBE = 32`.

*`tl.atomic_cas` takes no `mask`*, unlike `tl.atomic_add`. Resolved lanes cannot
be suppressed, so they must be aimed at a scratch address. Sizing that region
spans 20x: 10.8 ms at one shared slot, 1.8 ms at 256, 0.52 ms at 65,536.

**Corrected during this work.** The published figure was 73x; it is 17.9x at the
baseline cell. Most of the difference was a badly-sized workaround, not the
language (docs/SCRATCH-HYPOTHESIS.md). The smaller number is the stronger claim,
because it is device-independent — 17.9x and 17.9x on the two cards, within a
point or two per cell — where the old one halved between them.

### Triton's failure to scale, and its repair

Before the scratch fix, A5's allocate did not scale with SM count at all:
0.89x to 1.07x across a 2.33x SM difference, while A3 and A4 tracked the machine
at 1.8x to 2.1x. After: 1.76x to 2.07x, matching the other arms.

This is the sharpest expressiveness result in the project, because **the failure
was invisible on a single device.** One GPU reports 1.8 ms and looks reasonable.
Only the second card shows the kernel had stopped scaling, and only varying a
parameter of the workaround shows why.

### Rust's allocate gap: attributed, and the mechanism is a language cost

This took eleven hypotheses and a profiler, and it is the paper's cleanest
result because the answer was in none of the places a careful engineer looks
first.

Instruction counts never explained it. The kernels are at 1.02x static SASS
parity (520 against 528). A4 issues **12% fewer** dynamic compare-exchanges
(25,026 against 28,558), with fewer branches (63 against 72), fewer global
memory operations (6 against 11) and fewer registers (28 against 34), at
identical occupancy (84.27% against 84.30%) and with less warp divergence. It
did less work, more slowly.

Eliminated along the way, each tested rather than argued: register pressure,
`f32::clamp` NaN semantics, `read_volatile`, static instruction count, dynamic
compare-exchange count, memory-ordering scope, the shared block counter, the
`threadfence`, the publication store, the publication spin, and host-side launch
overhead.

Hardware counters found it in one run. `long_scoreboard` — the wait for a
long-latency memory operation — was the **only** stall reason that worsened, and
it worsened by more than the entire gap; every other reason favoured A4. The
cause was L1 residency: identical requests over identical sectors, but 1.70x the
sectors pushed through to L2, and an L1 hit rate of 28.9% against 55.9%.

Two loads. A4 read the probe key and the published block index with
`DeviceAtomic*::load(Relaxed)`; A3 reads both plainly. **A GPU-scope atomic load
must be coherent across SMs; NVIDIA's L1 is not coherent across SMs; so such a
load bypasses L1 by construction, on every call, however weak its ordering.**
Relaxed removes ordering, not coherence.

The index is almost always already published, so A4 paid an uncached load on the
common path where A3 pays a cached one and reserves the uncached load for when
it is genuinely waiting. Making both first reads plain — correct because the
*algorithm* rather than the memory model provides the guarantee, a stale key
costing one extra probe and a stale `-1` merely entering the wait loop — restores
every counter to parity: L1 hit rate 54.1%, L2 sectors 552,775 against A3's
568,128, `long_scoreboard` 11.06 against 11.54.

Allocate went from 1.63x to 1.33x at the baseline cell and 1.51x to 1.21x at
1.28M points; the full integrate path from 1.05–1.27x to 1.01–1.11x.

**Why this is the best expressiveness finding in the paper.** The construct that
looks safe and type-correct is the expensive one. Nothing in the source, the
PTX, the SASS, the instruction counts or the occupancy shows it. It is visible
only in where the data is *allowed to live*, and only to hardware counters.

### The residual, stated as such

Allocate is still 1.12–1.33x and that part is not attributed. The synthetic
sphere's 1.11x total is the small-scene outlier, not a language property; on real
data the figure is 1.01–1.03x.

## 2. Expressiveness: what each language cannot say

Each finding was hit while implementing the same algorithm, not constructed to
make a point.

**Triton cannot express an unbounded probe at any price.** The bound is a
`tl.static_range` trip count and therefore a compile-time constant. The
programmer must pick one, and a bound small enough to compile in seconds (32
took ~20 minutes and a 3.7x larger cubin) is small enough to **silently lose
data** at load factors CUDA C++ handles exactly: 13 blocks of 1,160 at load
factor 0.283, zero at the 0.035 every earlier measurement used. The loss was
unreported until this project added the accounting, because the pool-exhaustion
path had a drop counter and the probe-exhaustion path did not.

**Triton's missing `mask` on `atomic_cas` forces a data structure with no CUDA
counterpart**, and getting its size wrong destroys the kernel's ability to use
additional SMs while still reporting a plausible single-device time.

**Rust's scoped atomic load is correct and expensive, and nothing warns you.**
Detailed above. The generalisation: on a GPU, "make this read safe against
concurrent writers" and "make this read fast" are in tension in a way the type
system does not express, because coherence and caching are the same mechanism.

**cuda-oxide's scoped atomic load and store could not be called at all.** Every
call failed the libNVVM verifier under `--materialize-cubin`, which is any real
kernel, with an error naming an LLVM basic block rather than the offending line
of Rust. The same applied to any `AcqRel` or `SeqCst` ordering, via a rejected
LLVM `fence`. NVIDIA's own `atomics` example does not build in that mode.

Fixed and upstreamed: the PTX instructions have existed since sm_70, so the
three operations now lower to inline PTX (NVlabs/cuda-oxide#695, issue #696,
docs/CUDA-OXIDE-ATOMIC-LOAD.md). **This corrects an earlier claim in this
paper**, which said libNVVM's restrictions force Rust back onto volatile reads
and an explicit `threadfence`. They do not; the scoped API exists precisely to
avoid that. It was unusable, which is a different and more damning finding.

**A compiler crash with no source location** is materially worse than a type
error. A Triton atomic on a scalar pointer rather than a tensor of pointers
aborts the compiler with an MLIR assertion, `only integers and floats have a
bitwidth`.

## 3. Methodology, and a negative result

The measurement discipline caught roughly a dozen errors that would each have
produced a confident wrong number. The three worth a section produced a
*reversed or null* conclusion rather than a wrong magnitude, and two were caught
only after the wrong number had been written down:

* **A 40% workload asymmetry** (colour accumulation in three arms but not the
  fourth) made Rust look 2x faster than CUDA C++.
* **The device axis was silently fake.** A4's loader hardcodes device 0 and its
  driver-API context becomes current process-wide, so `cudaSetDevice(1)` was
  ignored by every arm. The first two-device sweep reported a clean null —
  identical performance across a 2.33x SM difference — while the second card sat
  idle at 8 W. Striking, publishable-looking, and wrong, with nothing anomalous
  in the timings.
* **Single-pass p50 is not reproducible.** One cell reported 0.047 ms in one pass
  and 0.060 in each of three reruns. No within-pass statistic separates it:
  flagging on p95/p50 fires on 8 of 10 cells and misses this one. The sweep now
  runs three passes and reports medians.

A fourth is worth listing because it was a *stage-level* number rather than an
arm-level one: the reported extraction time was 82–97% device-to-host readback
over PCIe links of gen2 x16 and gen1 x2, so its striking 6.9x "device
difference" was the motherboard. Real compute differs by 1.24x.

The lesson generalises: a language comparison's credibility rests on whether the
two sides did the same work on the same machine, and both turned out to be
things this harness had to *verify* rather than assume.

**The negative result.** Fusion backend choice is worth 0.2% to 2% of a stereo
reconstruction pipeline. That tells a practitioner where not to spend effort, and
it bounds the performance result honestly: the ranking is real and attributed,
and in this pipeline it is not what you should optimise first.

## What the depth stage is for

Context, not centrepiece. Two engines built and measured (13.4 ms and 157.2 ms),
enough to establish the budget share and bound the claim. The resolution
mismatch in docs/DEPTH-STAGE.md can stay open; the arms do not need real depth
to be compared.

## Design experiments that did not work, kept because they bound the space

**128-bit publication.** Publishing key and index in one `atom.cas.b128` would
remove the wait entirely. The instruction works in isolation; the algorithm does
not. The index must be reserved before the exchange, so a losing thread holds one
it cannot use, and at kernel start every thread targeting a block observes the
same empty slot before any exchange lands: 65,536 reservations against 1,160 real
blocks. This rules out the whole family — any scheme making the index part of the
published value needs speculative reservation.

**Slot-derived index.** Setting `block_idx = slot` removes the circularity
properly and is correct. It makes CUDA C++ 21–35% faster and Rust 29–32% faster,
at the cost of sizing the voxel pool to the table (~2x voxel memory) and changing
the shared data structure. **It does not close the language gap**: 1.64x becomes
1.71x at 320k and 1.53x becomes 1.39x at 1.28M. This corrected an earlier claim
that the publication spin was 75–83% of the gap — that measurement removed A4's
spin while A3 kept paying its own. The spin is a large *shared* cost.

## What this framing still needs

1. **A second architecture.** Both cards are sm_120, so the device axis varies
   machine width at fixed codegen and is explicitly not this. A cloud L4 (sm_89)
   is the cheapest way to show the ranking is not an accident of one chip. This
   is now the largest open item.
2. **The load-factor axis cannot exceed 0.283**, because the table is sized at
   twice the pool. Reaching the 0.7–0.9 regime where probe chains blow up needs
   table size decoupled from pool capacity. Until then the sharpest prediction of
   the Triton explanation stays untested.
3. **The residual allocate gap**, 1.12–1.33x, is not attributed.
4. **`MAX_PROBE = 32` measurements were taken at 256 scratch slots** and are
   therefore pessimistic by roughly 3.5x.
5. **Real data is one sequence.** Two cells from TartanAir V2 RetroOffice P000.
   They validated the synthetic sweep rather than contradicting it, but one
   environment is not a claim about real data in general.
