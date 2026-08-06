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
Batched, interleaved, exclusive GPU, p50 ms, 320k points at 0.01 m voxel:

| arm | allocate | update |
|---|---|---|
| A3 CUDA C++ | 0.028 | 0.249 |
| A4 Rust / cuda-oxide | 0.046 | 0.303 |
| A5 Triton | 2.040 | 0.352 |

Three findings, at three different levels of explanation.

**Triton's 73x on allocate is fully attributed, and it is a language cost.**
Both mechanisms are confirmed and they compound. No per-lane early exit means
the probe loop runs to a fixed bound, so cost is linear in that bound: 8.38x
for an 8x bound. And `tl.atomic_cas` takes no `mask`, so resolved lanes cannot
be masked off and must be aimed at a scratch address; pointing them at one
shared slot serialises the grid, 10.7 ms against 1.7 ms with a per-lane region,
5.9x. Replacing the CAS with a plain load leaves 0.090 ms, 113x less. This is
the cleanest result in the project: a measured slowdown traced to two specific
things the language cannot express.

**Rust's 1.22x on update is attributed to code generation.** It tracks a 1.62x
SASS instruction count, 504 against 312, concentrated in LOP3, FFMA, FSETP and
BRA. The sub-proportional runtime is consistent with a kernel partly bound by
its atomics. Refuted along the way: register pressure (A4 uses *fewer*
registers, 34 against 40, and neither spills), `f32::clamp` NaN semantics (zero
SASS change), and `read_volatile` on the hash key (byte-identical SASS).

**Rust's 1.64x on allocate is NOT attributed, and is reported as such.** The
two kernels are at 1.02x instruction parity, 528 against 520, so instruction
count explains the update gap and explicitly fails to explain this one. The
surviving candidate is memory-ordering scope: A4 issues six system-scope
strongly-ordered loads against A3's two, while issuing fewer atomics. That is
suggestive, not proof, and confirming it needs GPU counters.

Reporting a ratio without a mechanism is what makes most language comparisons
unciteable. Two of these three have one; the third says so.

### 2. Expressiveness: what each language cannot say

The qualitative half, and absent from the literature. Each finding was hit
while implementing the same algorithm, not constructed to make a point.

**Triton** cannot express per-lane early exit, and `tl.atomic_cas` takes no
`mask` unlike `tl.atomic_add`. Both are priced above; they are listed again
here because the pattern generalises beyond TSDF to any irregular kernel with
variable per-lane work.

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

The measurement discipline caught seven errors that would each have produced a
confident wrong number, including a 40% workload asymmetry that made Rust look
2x faster than CUDA, and a batching change that reversed an earlier
"A3 and A4 are identical" claim. These are worth a section, because a reader's
first question about any language comparison is whether the comparison was
fair, and because two of the seven errors would have produced a *reversed*
ranking rather than merely a wrong magnitude.

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

1. **More than one workload.** Every number comes from one sphere at 320k
   points. The comparison needs varied point counts and scene shapes, and at
   least one real TartanAir sequence, before a ratio is a property of the
   languages rather than of one scene.
2. **The allocate gap attributed.** Needs GPU performance counters, which need
   root: `NVreg_RestrictProfilingToAdminUsers=0` and a reboot.
3. **A second architecture.** One GPU makes every number a property of sm_120.
   A cloud L4 pass is the cheapest way to show the ranking is not an accident
   of one chip.
4. **Extraction measured properly, or explicitly excluded.** It is 1.825 ms,
   6.6x the whole integrate path, and currently shared and unoptimised. Either
   it joins the comparison or the paper says plainly that it does not.
