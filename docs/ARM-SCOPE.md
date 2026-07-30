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
