# Phase 0 spike results

All measured on nexxus (RTX 5070 Ti 16 GB + RTX 5060 8 GB, both sm_120,
CUDA 13.0.88, Ubuntu 24.10, gcc 14.2) on 2026-07-29 and 2026-07-30.

Purpose of Phase 0: answer the questions that would force a rewrite if
discovered later. Two of three spikes overturned a prediction in the roadmap.

| Spike | Question | Result |
|---|---|---|
| S1 | Can `cuda-oxide` (Rust to PTX) target sm_120 and express the TSDF primitives? | **PASS** |
| S2 | Can Triton express the irregular atomicCAS hash insertion? | **PASS**, with a measured cost |
| S3 | Does Open3D build as a C++ library here? | **PASS on CPU**, CUDA module blocked |

---

## S1: cuda-oxide on sm_120. PASSED.

**The documented blocker does not exist.** The `cuda-oxide` docs' prerequisites
table says "Ampere+ sm_80+" with advanced features "sm_80 through sm_100a",
which would exclude both local GPUs. The source disagrees:

- `crates/cargo-oxide/src/main.rs:830` lists `--arch sm_120  Blackwell (RTX 50 series)`
- `scripts/smoketest.sh` has a `blackwell-compile` target pinned to `sm_120a`
  that asserts `.target sm_120a` appears in the emitted PTX
- `scripts/smoketest.sh:237` maps `nvidia-smi` compute capability `12.0` to `sm_120`
- the intrinsics catalog gates Blackwell instructions on `sm_120a` / `sm_120f`

Reading the source rather than trusting the table removed a cloud dependency
from the critical path.

### Measured

Source: `spikes/s1_cuda_oxide/`. Two kernels mirroring
`openstrate-reconstruct-rs/vendor/libinfer/src/tsdf_hash.cu`:

```
--- Test 1: atomicCAS linear-probe hash insert ---
  512 threads racing for 64 distinct coords (8x contention)
  winners = 64 (expected 64)          unresolved threads = 0
  all duplicates agree on slot = true distinct slots used = 64
  => PASS
--- Test 2: atomicAdd SoA voxel accumulation ---
  total count = 512  weight = 256  sum_x = 512  sum_z = 1536   (all exact)
  every occupied slot has exactly 8 observations = true
  => PASS
=== SPIKE RESULT: PASS ===
RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

Emitted PTX: `.version 8.7`, **`.target sm_120`** (native, not JIT),
`.address_size 64`, containing `atom.acq_rel.gpu.global.cas.b32` and
`atom.global.add.f32` / `.u32`.

Test 1 deliberately validates the **claim protocol**, including the
lost-race-then-adopt path, rather than merely that CAS compiles.

Control: the shipped `atomics` example passes all 20 tests on sm_120.

### Consequences

- Develop and benchmark locally. The harness does not need to be remote-capable.
- Cloud L4 (sm_89) drops to a Phase 6 nice-to-have for the paper's second
  architecture axis. It is also the production target (`build.rs` defaults to
  `sm_89`).
- Rust's `compare_exchange` lowers to the same PTX instruction as the CUDA
  original, which is what makes the codegen comparison fair.

### Toolchain setup without root

`sudo` needs a password on nexxus, so the docs' apt route is unusable.
Everything installed into `$HOME`:

1. Ubuntu 24.10 ships LLVM 19 max; `cuda-oxide` needs 21+. Install the prebuilt
   `LLVM-21.1.8-Linux-X64.tar.xz` into `$HOME/.local/opt/llvm-21`.
   **It is needed only for the clang resource dir** (`lib/clang/21`); `llc`
   comes from the nightly's own bundled LLVM 22.1.2-rust. So the real
   dependency is the equivalent of `libclang-21-dev`, not a full toolchain.
2. `rustup toolchain install nightly-2026-04-03` with components
   `rust-src rustc-dev rust-analyzer clippy llvm-tools`. `rustc-dev` is easy to
   miss and required.
3. `cargo +nightly-2026-04-03 install --path crates/cargo-oxide`, then
   `cargo oxide setup`.
4. `PATH` prepends `$HOME/.local/opt/llvm-21/bin` and `/usr/local/cuda/bin`;
   set `LLVM_CONFIG_PATH` and `LIBCLANG_PATH` to the LLVM 21 tree.
5. `cargo oxide doctor` verifies everything and reports the GPU.

---

## S2: Triton hash-TSDF feasibility. PASSED BOTH PARTS.

**The roadmap predicted Triton could not express the irregular hash insertion.
That prediction was wrong.** Source: `spikes/s2_triton/`. Mirrors S1
test-for-test at identical contention.

```
--- Test B: atomicCAS linear-probe hash insert (the irregular part) ---
  winners = 64   unresolved = 0   duplicates agree = True
  distinct slots = 64   scratch slot intact = True    => PASS
--- Test A: atomicAdd SoA voxel accumulation (the regular part) ---
  count = 512  weight = 256.0  sum_x = 512.0  sum_z = 1536.0  => PASS
=== SPIKE RESULT: PASS ===
```

Triton emits `atom.global.acq_rel.gpu.cas.b32`, the **same PTX instruction
family** as cuda-oxide and the CUDA original. PTX `.target sm_120a`.

### The real finding is a control-flow tax, not a wall

1. **`tl.atomic_cas` takes no `mask`**, although `tl.atomic_add` does. Lanes
   that have already resolved cannot be masked off, so they must be redirected
   to a scratch slot holding a sentinel that can never satisfy the CAS. Costs a
   table slot and makes the kernel harder to reason about.
2. **No per-lane early exit.** The probe loop compiles to a rolled loop
   (PTX `bra` count 1, 438 PTX lines regardless of bound) that always runs the
   full trip count. CUDA and cuda-oxide `break` per thread on success.

| MAX_PROBE | latency | registers |
|---|---|---|
| 4 | 7.9 us | 40 |
| 8 | 12.3 us | 38 |
| 64 (same bound as the CUDA original) | 65.5 us | 38 |

Linear in the bound, an **8.3x penalty from 4 to 64**, on a workload where
nearly every lane resolves within one or two probes. Triton pays worst case;
CUDA pays actual.

For comparison, the regular accumulation kernel is Triton's natural ground:
24 registers, 0 spills, 9.3 us.

A measured penalty with an identified mechanism is a better result than a
feasibility shrug, and it is useful to anyone evaluating Triton for volumetric
fusion or any open-addressed GPU hash table.

### Gotcha

`tt.atomic_cas` verifies the cmp operand dtype against the pointee dtype
exactly. Passing a Python `constexpr` int fails with
`'tt.atomic_cas' op failed to verify that ptr type matches cmp type`.
Materialise the sentinel as `tl.full((BLOCK,), X, tl.int32)`.

### Consequence

Splitting insertion from integrate is **no longer forced** by Triton's limits,
though still worth doing so all arms share one allocation path and the
comparison measures the update rather than allocation-order luck. It also
unlocks a bonus data point: run the Triton arm both ways, insertion-shared and
Triton-does-everything, and price the tax end to end.

---

## S3: Open3D C++ from source. CPU PASSES. CUDA module blocked.

**Open3D 0.19 is not CUDA 13 clean.** Configure succeeded with
`-DCMAKE_CUDA_ARCHITECTURES=120` (Open3D's own defaults stop at sm_90, but it
honours a user-provided value and nvcc 13 accepts sm_120). The build then hit
four separate CUDA 13 incompatibilities, each silent until the previous was
cleared.

### Fixed: three issues in `stdgpu` (pinned by Open3D to a 2023 commit)

See `spikes/s3_open3d/`.

1. **`thrust/version.h` is gone.** CUDA 13 moved Thrust into the CCCL subtree
   at `include/cccl/thrust/`. `Findthrust.cmake`'s regex returned empty,
   producing `Found unsuitable version "ERROR.ERROR.ERROR"`. Fixed by pointing
   `THRUST_INCLUDE_DIR` at the CCCL root. The version is actually 3.0.1, far
   above the 1.13.1 floor.
2. **CUB and libcu++ paths.** Same relocation, so `LIBCUDACXX_INCLUDE_DIR` and
   `CUB_INCLUDE_DIR` also need the CCCL root. All three resolve to one directory.
3. **`cudaDeviceProp::clockRate` removed.** CUDA 13 dropped the deprecated
   field. `stdgpu` reads it only to pretty-print device info, so it is replaced
   with `cudaDeviceGetAttribute(cudaDevAttrClockRate)`. Cosmetic, no behaviour
   change. Applied as an idempotent `PATCH_COMMAND` script rather than an edit
   to the extracted tree, because `ExternalProject` re-extracts sources and an
   in-tree edit vanishes silently on a clean rebuild.

With these, `stdgpu` builds clean at 100%.

### Not fixed: Open3D's own CUDA hashmap vs Thrust 3

`cpp/open3d/core/hashmap/CUDA/CreateCUDAHashBackend.cu` fails against Thrust
3.0.1 with errors including:

```
initial value of reference to non-const must be an lvalue
reference to void is not allowed
no suitable constructor exists to convert from "unsigned int *" to
  thrust::pointer<..., cuda_cub::tag, ...>
```

This is Thrust 3 API churn in `device_vector` iterator and pointer semantics,
inside Open3D core rather than a third-party dependency, and specifically in
the GPU hashmap that `VoxelBlockGrid` depends on. Not a small patch.

### Options

1. **CPU-only Open3D** (`-DBUILD_CUDA_MODULE=OFF`). **DONE, clean build, 0
   errors.** Installed to `$HOME/.local/opt/open3d-cpu`. Verified from C++
   against the installed library (`spikes/s3_open3d/acceptance/`):

   ```
   Open3D 0.19.0
     unique block coords: 3700
     hashmap size after integrate: 3700
     mesh: 463761 vertices, 924800 triangles
     sphere r=1.00 vs r=1.02: mean surface dist 0.0460, hausdorff 0.1216
   VoxelBlockGrid (A1)        : PASS
   Mesh comparison (P1 gate)  : PASS
   ```

   Both things the project needs are confirmed working: the A1 arm on CPU, and
   the mesh-comparison API the Phase 1 correctness gate depends on regardless of
   whether A1 ships.

   Trap: `ExtractTriangleMesh` defaults to `weight_threshold = 3.0`, so a single
   integration extracts zero vertices. That looks exactly like a broken TSDF and
   is not one.
2. **Newer Open3D** from `main`, which may have Thrust 3 support.
3. **Use the Python wheel** for the Open3D CUDA arm and the C++ library for CPU.
   Mixed-language, but the wheel ships a working CUDA build.

### Consequence for the plan

This is the strongest argument yet for treating the Open3D arm as a **droppable
baseline** (fallback item 3 in the roadmap) rather than a load-bearing arm. It
also carries forward to the cloud L4: any machine used for benchmarking needs
the same patches, or a CUDA 12 toolchain.

Note the irony worth a sentence in the paper: the Open3D component that will
not build is its GPU hash map, which is the same data structure this project
implements five ways.
