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

## The claim

**What does it cost to express an irregular, atomics-heavy GPU workload in CUDA
C++, in Rust, and in Triton?**

A hash-blocked TSDF integrate path is a good vehicle for the question because
it is the opposite of the tiled dense linear algebra these tools are usually
compared on: an open-addressed hash table with compare-exchange insertion,
per-lane variable probe depth, scattered atomic accumulation, and no regular
tile structure anywhere.

Three contributions, in decreasing order of confidence:

### 1. Expressiveness: what each language cannot say

This is the strongest material because it is qualitative, reproducible, and
absent from the literature. Each finding below was hit while implementing the
same algorithm, not constructed to make a point.

**Triton** cannot express per-lane early exit. A probe loop must run to a fixed
bound with a `done` mask, so every lane pays worst-case probe depth. Measured:
cost is linear in the bound, 8.38x for an 8x bound. And `tl.atomic_cas` takes
no `mask`, unlike `tl.atomic_add`, so lanes that have already resolved cannot
be masked off; they must be aimed at a scratch address instead. Pointing them
all at one slot serialises the grid: 10.7 ms with a shared slot against 1.7 ms
with a per-lane region, a 5.9x difference. Both effects compound, and neither
is visible from a throughput table.

**cuda-oxide** (Rust to PTX) is constrained by libNVVM rather than by Rust.
libNVVM rejects atomic loads and stores outright, accepting only read-modify-
write, and rejects LLVM `fence`, so Acquire/Release orderings on RMW operations
are unusable. Both restrictions force the code back onto exactly the
constructs CUDA C++ uses: volatile reads and an explicit `threadfence`. The
default LLVM path accepts what libNVVM rejects, so code can build in one mode
and fail under `--materialize-cubin`.

**A compiler crash with no source location** is a materially worse developer
experience than a type error. Issuing a Triton atomic on a scalar pointer
rather than a tensor of pointers aborts the compiler with an MLIR assertion,
`only integers and floats have a bitwidth`.

### 2. Performance, with attribution where it exists

Integrate path only; extraction is shared across GPU arms (docs/ARM-SCOPE.md).
Batched, interleaved, exclusive GPU, p50 ms, 320k points at 0.01 m voxel:

| arm | allocate | update |
|---|---|---|
| A3 CUDA C++ | 0.028 | 0.249 |
| A4 Rust / cuda-oxide | 0.046 | 0.303 |
| A5 Triton | 2.040 | 0.352 |

Attribution so far, in docs/A3-A4-GAP.md: A4's 1.22x on update tracks a 1.62x
SASS instruction count. Register pressure is refuted (A4 uses fewer registers
and neither spills), `f32::clamp` is refuted, and `read_volatile` on the hash
key is refuted (identical SASS). The 1.64x on allocate remains unattributed at
1.02x instruction parity, with memory-ordering scope the surviving candidate.

Reporting a ratio without a mechanism is what makes most language comparisons
unciteable. Any number that stays unattributed should be reported as such.

### 3. Methodology, and a negative result on budget share

The measurement discipline caught seven errors that would each have produced a
confident wrong number, including a 40% workload asymmetry that made Rust look
2x faster than CUDA, and a batching change that reversed an earlier
"A3 and A4 are identical" claim. These are worth a section, because a reader's
first question about any language comparison is whether the comparison was
fair.

The dead crossover becomes a secondary result rather than being buried: fusion
backend choice is worth between 0.2% and 2% of a stereo reconstruction
pipeline. That tells a practitioner where not to spend effort, which is what a
latency-budget paper should do.

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
