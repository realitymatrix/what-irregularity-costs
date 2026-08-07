# What an "arm" is, and what is held constant

The comparison is only meaningful if the arms differ in the dimension being
measured and in nothing else. This records where the boundary sits.

## Arms differ in the INTEGRATE path

Every GPU arm implements two passes and nothing more:

    allocate_blocks(batch)   claim hash slots for the blocks the batch touches
    update_voxels(batch)     accumulate weighted sums into those blocks

| Arm | allocate | update |
|---|---|---|
| A3 cuda | CUDA C++ | CUDA C++ |
| A4 rust-cuda | Rust via cuda-oxide | Rust via cuda-oxide |
| A5a triton-shared | CUDA C++ (shared with A3) | Triton |
| A5b triton-full | Triton | Triton |

**Surface extraction is shared** across the GPU arms, using A3's marching
tetrahedra. Two reasons:

* The claim under test is about fusion throughput. Integrate runs once per
  frame; extraction runs once per sequence.
* Three independent extractors would differ for reasons unrelated to the
  languages, and those differences would land in the same numbers as the effect
  being measured.

A2 (CPU) necessarily has its own extractor. It is on a different device class
and is reported on its own axis anyway.

## Held constant across arms

* **Input.** World-space points, produced once by the loader, so pose
  composition cannot differ.
* **Hash.** Same multiply-xor mix and same table size. A different mix changes
  probe sequences, which changes which blocks share cache lines, and that would
  read as a language difference.
* **Key layout.** One packed 64-bit key per slot, so a single compare-exchange
  publishes the whole coordinate.
* **Allocation gate.** Both passes cull fully occluded voxels (`sdf < -trunc`).
  Applying it in only one pass leaves blocks that never receive a contribution:
  the meshes still match, but block counts diverge.
* **Accumulation.** Weighted sums, not running means. Sums commute, so plain
  atomic adds suffice and no read-modify-write exists to race.
* **SDF sample point.** Voxel centre, `(v + 0.5) * voxel_size`.
* **Architecture.** Native `sm_120` for every arm. No arm JITs from PTX while
  another runs native SASS.
* **Launch driver.** One Rust driver for every arm, so no arm pays a
  language-runtime cost the others do not.

## Verified equivalence, not assumed

Each arm passes the analytic tests (tier 1) and is cross-checked against the
others (tier 3). A3 and A4 emit the same PTX instruction family for the two
operations that matter:

    A3 (CUDA C++)          atomicCAS on u64  -> atom.acq_rel.gpu.global.cas.b64
    A4 (Rust/cuda-oxide)   compare_exchange  -> atom.acq_rel.gpu.global.cas.b64
    A5 (Triton)            tl.atomic_cas     -> atom.global.acq_rel.gpu.cas.b32

so the comparison measures code generation rather than a difference in the
primitive selected. (A5 is b32 because its table is an i32 index table; see
docs/TRITON-ABI.md.)

---

# Decision: extraction is EXCLUDED from the language comparison

Decided 2026-08-07, on evidence from `bench/probe_extract.cu`. Recorded here
rather than left open, because "we did not compare it" and "we compared it and
it did not matter" are different claims and a reader is entitled to know which.

## Three reasons, in order of how decisive they are

**1. As built, it compares nothing.** Marching tetrahedra exists once, in CUDA
C++, and A3, A4 and A5 all call the same implementation. Every "extraction"
number in this project is arm A3 measured three times. Making it a real
comparison means writing marching tets a second time in Rust and a third time
in Triton, which is a substantial piece of work.

**2. That work would not extend the thesis.** The paper is about irregular,
atomics-heavy workloads: data-dependent per-lane trip counts, compare-exchange
insertion, contended scatter. Marching tets has none of that. It is one CUDA
block per pool block, one thread per voxel, a small constant table lookup, and
a single-counter `atomicAdd` for output compaction. It is a regular kernel, and
the comparison already contains one: the `update` stage, where the three arms
land within 1.0x to 2.5x of each other. A second regular kernel would add a
row, not an argument.

**3. The number previously reported was not a kernel time.** This is the part
that forced the decision rather than merely permitting it.

`CudaVolume::extract_mesh` does four things in one call: a 96 MiB `cudaMalloc`,
the kernel, a device-to-host copy of every vertex produced, and the matching
`cudaFree`. Only the kernel is language-comparable. Splitting them by running
extraction at an impossible `min_weight`, so the kernel still sweeps every block
but no vertex survives the readback:

| | RTX 5070 Ti | RTX 5060 |
|---|---|---|
| full extract | 1.792 ms | 12.766 ms |
| kernel + allocation only | 0.319 ms | 0.396 ms |
| device-to-host readback | 1.472 ms (82.2%) | 12.370 ms (96.9%) |
| implied host bandwidth | 12.85 GiB/s | 1.53 GiB/s |

**82% to 97% of the measurement was PCIe.** The two cards are not on comparable
links: device 0 negotiates gen2 x16, device 1 gen1 x2, and the 8.4x measured
bandwidth ratio matches that topology. The striking 6.9x device difference that
prompted this investigation was the motherboard, not the GPU. Actual compute
differs by 1.24x, which is unremarkable.

Two previously published statements are therefore wrong and are corrected here:

* *"Extraction is 1.825 ms"* — that is a transfer time. The kernel and its
  allocation are 0.319 ms on the same card.
* *"6.6x the whole integrate path"* — no. At 0.319 ms against an integrate path
  of roughly 0.28 ms (0.028 allocate + 0.251 update), extraction is comparable
  to integrate, not several times it.

## What the paper says instead

Extraction is out of scope, and the reason given is (1) and (2), not the
measurement artefact. The artefact is worth one sentence in the methodology
section as an example of a stage-level number that turned out to be a bus
measurement, which is a general hazard when a pipeline stage ends in a readback.

If extraction is ever brought in, the prerequisites are: hoist the scratch
allocation out of the measured region, return device pointers rather than
copying to host, and implement the kernel in all three languages. Until all
three are done there is nothing to compare.

## Harness behaviour

`bench_arms` no longer measures extraction by default. `--extract` opts in, and
the emitted rows are labelled `extract-a3-only` so a CSV can never be read as a
three-way comparison.
