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

**Rust is 1.23x slower on update at the baseline, and at parity for large
scenes.** The baseline gap tracks a 1.62x SASS instruction count, 504 against
312, concentrated in LOP3, FFMA, FSETP and BRA. But the ratio falls to 0.97–1.02x
on the largest cells, where Rust is at or slightly ahead of CUDA C++. That is
consistent and worth stating plainly: instruction count sets the gap while the
kernel is issue-bound, and stops mattering once the working set makes it
memory-bound. Refuted along the way: register pressure (A4 uses *fewer*
registers, 34 against 40, neither spills), `f32::clamp` NaN semantics (zero SASS
change), and `read_volatile` on the hash key (byte-identical SASS).

**Rust's allocate gap is NOT attributed, and the sweep made it harder.** It
ranges 1.24x to 4.06x with the workload. The two kernels are at 1.02x
instruction parity, 528 against 520, so instruction count explains the update
gap and explicitly fails here. The surviving candidate was memory-ordering
scope, six system-scope strongly-ordered loads against A3's two — but scope is a
static property of the generated code and cannot vary with workload, so it
cannot be the whole story. Confirming anything here needs GPU counters.

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
   rather than confirming them. Still missing: one real TartanAir sequence, so
   every scene remains synthetic, and the `loadfactor` axis cannot exceed 0.283
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
4. **Extraction measured properly, or explicitly excluded.** It is 1.85 ms on
   the 5070 Ti, 6.6x the whole integrate path, shared and unoptimised. The sweep
   added a reason to decide rather than defer: it costs 12.72 ms on the 5060, a
   6.9x device difference against a 2.33x SM ratio, which is nothing like how
   the integrate stages behave and is unexplained.
5. ~~**The scratch-region hypothesis tested.**~~ **Done and confirmed**,
   docs/SCRATCH-HYPOTHESIS.md, and the full sweep re-run at 65,536 slots. It
   cost the headline ratio: 73x became 18.9x. Still open: the `MAX_PROBE = 32`
   measurements were taken at 256 slots and are therefore pessimistic by roughly
   3.5x too.
