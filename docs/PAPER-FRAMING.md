# Paper framing

Reframed 2026-07-30, after the depth measurements showed the original crossover
claim is not supported. See docs/DEPTH-STAGE.md for the numbers that forced it.

## What changed and why

The project was scoped around a crossover: with a heavyweight depth model the
fusion backend is irrelevant, but with a real-time model fusion becomes the
bottleneck and backend choice decides frame rate. Measured across an 11.8x
depth range, fusion is 0.18% to 2.0% of the pipeline and never approaches being
the bottleneck. The claim is dead, and no caveat rescues it: the fastest
available depth model, at fewer pixels than the dataset's native resolution,
still costs 48x the fastest fusion arm.

The language comparison was always the more novel half. It is now the paper.
The performance comparison is not diminished by this; it is promoted. It stops
being an input to a budget argument and becomes the result itself, which is
also what forces the attribution work: a ratio that was only ever feeding a
crossover could be reported bare, and a ratio that IS the paper cannot.

## The claim

**What does it cost to express an irregular, atomics-heavy GPU workload in CUDA
C++, in Rust, and in Triton?**

A hash-blocked TSDF integrate path is a good vehicle for the question because
it is the opposite of the tiled dense linear algebra these tools are usually
compared on: an open-addressed hash table with compare-exchange insertion,
per-lane variable probe depth, scattered atomic accumulation, and no regular
tile structure anywhere.

The question is answered on two axes, and both are load-bearing:

* **Performance.** The three arms are not equally fast on the same algorithm
  over the same memory, and the differences are large enough to matter to
  someone choosing a language. Measuring them, and attributing them to a
  mechanism, is the primary result.
* **Expressiveness.** What each language cannot say. This is not a separate
  paper: several of the performance gaps ARE expressiveness costs made
  numerical, which is what lets the measurement be explained rather than just
  reported.

### 1. Performance, and why each arm lands where it does

Integrate path only; extraction is shared across GPU arms (docs/ARM-SCOPE.md).
Batched, interleaved, exclusive device, medians of three passes, 320k points on
a 0.5 m sphere at 0.01 m voxel, 65,536 scratch slots. This is the `base` cell of
a 13-cell sweep (docs/WORKLOAD-SWEEP.md); the ranges beside it are that cell's
value against the full sweep's spread on the same device.

**RTX 5070 Ti, 70 SMs**, p50 ms:

| arm | allocate | vs A3 | sweep range | update | vs A3 | sweep range |
|---|---|---|---|---|---|---|
| A3 CUDA C++ | 0.0279 | — | — | 0.2508 | — | — |
| A4 Rust / cuda-oxide | 0.0460 | 1.65x | 1.24–4.06x | 0.3081 | 1.23x | 0.97–1.24x |
| A5 Triton | 0.5277 | 18.9x | 9.1–26.2x | 0.3513 | 1.40x | 1.13–2.53x |

**RTX 5060, 30 SMs**, same cell:

| arm | allocate | vs A3 | update | vs A3 |
|---|---|---|---|---|
| A3 CUDA C++ | 0.0541 | — | 0.4558 | — |
| A4 Rust / cuda-oxide | 0.0783 | 1.45x | 0.4810 | 1.06x |
| A5 Triton | 0.9642 | 17.8x | 0.5595 | 1.23x |

**No ratio in this project may be quoted without naming its workload and its
device.** That is not a caveat, it is a finding: the single-cell numbers this
document previously carried were wrong by up to 3.9x, and two of them were wrong
in a way that reversed the interpretation.

Four findings, at four different levels of explanation.

**Triton's allocate gap is 9x to 26x, about 19x at the baseline cell, and it is
attributed.** Two mechanisms, both confirmed, both consequences of things the
language cannot express, and they compound.

*No per-lane early exit*: the probe loop runs to a fixed bound, so cost is
linear in that bound, measured at 8.38x for an 8x bound and confirmed again at
`MAX_PROBE = 32`. *`tl.atomic_cas` takes no `mask`*: resolved lanes cannot be
suppressed and must be aimed at scratch, and the sizing of that scratch region
spans 20x (10.8 ms at one slot, 1.8 ms at 256, 0.52 ms at 65,536).

The published figure was 73x. It is 18.9x. **Most of the difference was a
badly-sized workaround, not the language** (docs/SCRATCH-HYPOTHESIS.md). The
smaller number is nonetheless the stronger claim, because it is
device-independent — 18.9x and 17.8x on two cards, and within a point or two per
cell across the sweep — where the old one halved between them.

**Triton's allocate scales with the machine, once the workaround is sized
right.** 1.76x to 2.07x across a 2.33x SM difference, against A3's 1.60x to
2.10x. Before the fix it was 0.89x to 1.07x: the kernel could not use additional
SMs at all. This is the sharpest expressiveness result in the project, because
the failure was invisible on a single device — one GPU reports 1.8 ms and looks
reasonable.

**Rust is within 1-5% of CUDA C++ on the full integrate path.** Total
allocate + update, medians of three passes, both cards:

| cell | A3 | A4 | ratio |
|---|---|---|---|
| TartanAir warm, 5070 Ti | 0.3878 | 0.3960 | **1.02x** |
| TartanAir cold, 5070 Ti | 0.4726 | 0.4863 | **1.03x** |
| TartanAir cold, 5060 | 0.9041 | 0.9154 | **1.01x** |
| plane-320k, 5070 Ti | 0.3633 | 0.3791 | 1.04x |
| pts-1280k, 5070 Ti | 0.9761 | 1.0254 | 1.05x |
| baseline sphere, 5070 Ti | 0.2781 | 0.3077 | 1.11x |

**This is after the L1 fix and supersedes every earlier figure.** The gap was
1.05x to 1.27x before; it is 1.01x to 1.11x now, and the remaining spread is
the small-synthetic-scene effect, not a language property. On real data Rust is
within 1-3% of hand-written CUDA C++. The baseline gap tracks a 1.62x SASS instruction count, 504 against
312, concentrated in LOP3, FFMA, FSETP and BRA. But the ratio falls to 0.97–1.02x
on the largest cells, where Rust is at or slightly ahead of CUDA C++. That is
consistent and worth stating plainly: instruction count sets the gap while the
kernel is issue-bound, and stops mattering once the working set makes it
memory-bound. Refuted along the way: register pressure (A4 uses *fewer*
registers, 34 against 40, neither spills), `f32::clamp` NaN semantics (zero SASS
change), and `read_volatile` on the hash key (byte-identical SASS).

**Rust's allocate gap IS attributed, and the mechanism is a language cost.**
It took eleven hypotheses and a profiler. Instruction counts never explained it:
the kernels are at 1.02x static parity and A4 issues 12% *fewer* dynamic
compare-exchanges, with fewer branches, fewer memory operations and fewer
registers, at identical occupancy.

Hardware counters found it in one run. `long_scoreboard`, the wait for a
long-latency memory operation, was the only stall reason that worsened, and it
worsened by more than the whole gap. The cause was L1 residency: identical
requests over identical sectors, but 1.70x the sectors pushed through to L2, and
an L1 hit rate of 28.9% against 55.9%.

Two loads. A4 read the probe key and the published block index with
`DeviceAtomic*::load(Relaxed)`; A3 reads both plainly. **A GPU-scope atomic load
must be coherent across SMs, and NVIDIA's L1 is not, so it bypasses L1 by
construction.** The index is almost always already published, so A4 was paying
an uncached load on the common path where A3 pays a cached one and reserves the
uncached load for when it is actually waiting.

Making both first reads plain, which is correct because the algorithm rather
than the memory model provides the guarantee, restores every counter to parity
(L1 hit rate 54.1%, L2 sectors 552,775 against A3's 568,128, `long_scoreboard`
11.06 against 11.54) and takes allocate from 1.63x to 1.33x at the baseline cell
and 1.51x to 1.21x at 1.28M points on the wide card.

This is the paper's cleanest expressiveness result, because the safe-looking
construct is the expensive one and nothing short of hardware counters would
show it. The residual 1.1x to 1.3x on allocate is not attributed.

Reporting a ratio without a mechanism is what makes most language comparisons
unciteable. Three of these four have one; the fourth says so.

### 2. Expressiveness: what each language cannot say

The qualitative half, and absent from the literature. Each finding was hit
while implementing the same algorithm, not constructed to make a point.

**Triton** cannot express per-lane early exit, and `tl.atomic_cas` takes no
`mask` unlike `tl.atomic_add`. Both are priced above; they are listed again
here because the pattern generalises beyond TSDF to any irregular kernel with
variable per-lane work.

Two consequences that a throughput table alone would not surface. First, the
probe bound is a `tl.static_range` trip count and therefore a compile-time
constant, so **an unbounded probe is inexpressible at any price**: the
programmer must pick a bound, and a bound small enough to compile in seconds
(32 took ~20 minutes and a 3.7x larger cubin) is small enough to silently lose
data at load factors CUDA C++ handles exactly. Second, the scratch workaround
the missing `mask` forces has a sizing parameter with no counterpart in the
CUDA implementation, and getting it wrong destroys the kernel's ability to use
additional SMs while still reporting a plausible single-device time.

**cuda-oxide** (Rust to PTX) is constrained by libNVVM rather than by Rust.
libNVVM rejects atomic loads and stores outright, accepting only read-modify-
write, and rejects LLVM `fence`, so Acquire/Release orderings on RMW operations
are unusable. Both restrictions force the code back onto exactly the
constructs CUDA C++ uses: volatile reads and an explicit `threadfence`. The
default LLVM path accepts what libNVVM rejects, so code can build in one mode
and fail under `--materialize-cubin`. The irony is worth stating: the safety
argument for Rust on the GPU is weakened when the backend removes the ordering
primitives the safe abstractions are built on.

**A compiler crash with no source location** is a materially worse developer
experience than a type error. Issuing a Triton atomic on a scalar pointer
rather than a tensor of pointers aborts the compiler with an MLIR assertion,
`only integers and floats have a bitwidth`.

### 3. Methodology, and a negative result on budget share

The measurement discipline has caught roughly a dozen errors that would each
have produced a confident wrong number. Three of them would have produced a
*reversed or null* conclusion rather than merely a wrong magnitude, and two were
caught only after the wrong number had already been written down:

* **A 40% workload asymmetry** (colour accumulation in three arms but not the
  fourth) made Rust look 2x faster than CUDA C++.
* **The device axis was silently fake.** Arm A4's loader hardcodes device 0 and
  its driver-API context becomes current process-wide, so `cudaSetDevice(1)` was
  ignored by every arm. The first two-device sweep reported a clean null —
  identical performance across a 2.33x SM difference — while the second card sat
  idle at 8 W. That is a striking, publishable-looking result with a ready
  explanation, and nothing in the timings looked anomalous.
* **Single-pass p50 is not reproducible** for this stage. One cell reported
  0.047 ms in one pass and 0.060 in each of three reruns, a 30% swing. No
  within-pass statistic separates it: flagging on p95/p50 fires on 8 of 10 cells
  while missing this one. The sweep now runs three passes and reports medians.

The lesson generalises past this project, which is why it is worth a section: a
language comparison's credibility rests entirely on whether the two sides were
doing the same work on the same machine, and both of those turned out to be
things this harness had to *verify* rather than assume.

The dead crossover becomes a secondary result rather than being buried: fusion
backend choice is worth between 0.2% and 2% of a stereo reconstruction
pipeline. That tells a practitioner where not to spend effort, which is what a
latency-budget paper should do. It also bounds the performance result honestly:
the ranking is real and attributable, and in this particular pipeline it is
not what you should optimise first.

## What the depth stage is now for

Context, not the centrepiece. Two engines are built and measured
(13.4 ms and 157.2 ms), which is enough to establish the budget share and to
bound the claim. No further depth work is needed for the paper: the resolution
mismatch decision in docs/DEPTH-STAGE.md can stay open, because the arms do not
need real depth to be compared.

## What this framing still needs

1. ~~**More than one workload.**~~ **Done**, docs/WORKLOAD-SWEEP.md: 13 cells
   across four axes, three passes, both local cards. It changed the conclusions
   rather than confirming them. **Real data added 2026-08-07**: two TartanAir
   V2 RetroOffice cells, cold and warm. They validated the synthetic sweep
   (real ratios land inside the synthetic range and within a fraction of a
   point of the baseline), showed Rust's update gap to be a small-scene
   artefact (1.00–1.02x on real data against 1.23x on the sphere), and reached
   the regime the `loadfactor` axis could not: in the warm volume a live
   pipeline actually runs in, CUDA and Rust take the early exit and get 1.3–1.4x
   faster while Triton does not move at all, widening the gap from 17.2x to
   23.7x. Still open: the `loadfactor` axis cannot exceed 0.283
   because the table is sized at twice the pool. Reaching the 0.7–0.9 regime
   where probe chains actually blow up needs table size decoupled from pool
   capacity, and until then the sharpest prediction of the Triton explanation
   stays untested.
2. **The allocate gap attributed.** Needs GPU performance counters, which need
   root: `NVreg_RestrictProfilingToAdminUsers=0` and a reboot. The sweep
   weakened the surviving hypothesis: the gap varies with workload, and
   memory-ordering scope is a static property of the generated code that does
   not.
3. **A second architecture.** Both local cards are sm_120, so the device axis
   varies machine width at fixed codegen and is explicitly not this. A cloud L4
   (sm_89) is the cheapest way to show the ranking is not an accident of one
   chip.
4. ~~**Extraction measured properly, or explicitly excluded.**~~ **Decided:
   EXCLUDED**, docs/ARM-SCOPE.md. It exists once, in CUDA C++, and all three GPU
   arms call it, so every extraction number was arm A3 measured three times.
   Bringing it in means writing marching tets in Rust and Triton, and that work
   would not extend the thesis: it is a regular kernel, and the comparison
   already has one in the update stage.

   Investigating it also corrected two published statements. The 6.9x device
   difference was PCIe topology, not compute: 82% of the measurement on the
   5070 Ti and 97% on the 5060 was the device-to-host readback, over links that
   are gen2 x16 and gen1 x2 respectively. Kernel plus allocation is 0.319 ms and
   0.396 ms, a 1.24x difference. So "extraction is 1.825 ms" was a transfer
   time, and "6.6x the whole integrate path" was wrong: at 0.319 ms against
   roughly 0.28 ms it is comparable to integrate.
5. ~~**The scratch-region hypothesis tested.**~~ **Done and confirmed**,
   docs/SCRATCH-HYPOTHESIS.md, and the full sweep re-run at 65,536 slots. It
   cost the headline ratio: 73x became 18.9x. Still open: the `MAX_PROBE = 32`
   measurements were taken at 256 slots and are therefore pessimistic by roughly
   3.5x too.
