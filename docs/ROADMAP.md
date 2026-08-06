# ROADMAP: stereo-depth-to-TSDF fusion, multi-backend latency study

Written 2026-07-29 on nexxus, superseding the phasing in `PROJECT-tsdf-backends-scope.md`
and `HANDOFF-tsdf-backends-project.md`. Those documents were written before the source
tree, the GPUs, and the dataset had been inspected. Everything below marked VERIFIED was
measured on this machine today; everything marked ASSUMPTION still needs confirming.

Gates closed: P1.1 (production C++ authorship), P1.2 (Triton), P2.1 (publication),
plus P1.5 (Open3D) as a side effect.

---

## 0. What changed since the scope doc

Read this section before following the old phasing, because five of its premises moved.

### 0.1 Scope decisions taken 2026-07-29

The arm count grew from three to five. TSDF fusion arms are now:

| # | Arm | Authored by | Purpose |
|---|-----|-------------|---------|
_(A0, the existing Rust/CUDA hash TSDF as a vendored golden reference, was
**removed 2026-07-30**. No code is reused from OSN; it is a rough reference for
algorithm and API shape only. See section 0.5 for what replaces it.)_

| A1 | Open3D `VoxelBlockGrid` (C++) | Third party | External baseline. Answers "why not just use Open3D" |
| A2 | Our own C++ TSDF, CPU, C++17/20 | Petr, new | **P1.1 evidence.** Pure C++, no CUDA behind it |
| A3 | Our own CUDA C++ TSDF | Petr, new | Kernel authorship, the performance reference point |
| A4 | Our own Rust CUDA TSDF via `cuda-oxide` | Petr, new | Rust-to-PTX vs nvcc codegen |
| A5a | Triton fusion, insertion shared with A3 | Petr, new | **P1.2 evidence.** Clean update-only comparison vs A3 |
| A5b | Triton fusion, insertion in Triton too | Petr, new | Prices the control-flow tax end to end |

**A5 ships as two variants, decided 2026-07-30.** Spike S2 showed the irregular
CAS insertion IS expressible in Triton, so the second variant became available.
A5a isolates the regular per-block update; A5b adds Triton's own insertion. The
**delta between them is a direct measurement** of the control-flow tax
(no per-lane early exit, no mask on `atomic_cas`) rather than an inference from
microbenchmarks. Seven arms total.

### 0.2 One contradiction to resolve, resolved in favour of the scope doc

Mid-session the instruction was "Triton will drive the inference", which reads as Triton
sitting in the depth path. The pasted scope doc says the opposite, explicitly and with a
better argument: *"Triton's role is the fusion backend, not depth pre/post-processing.
Rectification and normalisation kernels are memory-bound and prove nothing."*

**RESOLVED 2026-07-30, in favour of the scope doc.** Triton is arm A5, a fusion backend
competing directly with A3. Depth runs on TensorRT via **Rust libinfer**, not Triton and
not torch.compile. Rationale beyond the scope doc's: a Triton/Inductor depth stage would
be slower than the existing TensorRT FP16 path, and a *slower* depth stage pushes the
crossover the wrong way by making fusion look less like the bottleneck. It would also
discard the FP16-safe-set and custom-operator work that is the project's bonus gate.

### 0.3 VERIFIED findings that change the plan

1. **Extraction boundary is already clean.** `vendor/libinfer/src/tsdf_hash.h` (337 lines)
   and `tsdf.h` (159 lines) are pure `extern "C"`: scalars, raw pointers, and an opaque
   `typedef struct TsdfHash TsdfHash`. No ARCore, no session state, no Rust types, no
   TensorRT. Includes are only `cuda_runtime.h` plus four C headers. The API takes
   **world-space points**, so pose composition never enters the volume. These files
   compile standalone under CMake essentially unmodified.

2. **The TSDF is not in the `cxx::bridge`.** It builds as a separate `cc::Build.cuda(true)`
   static lib (`tsdf_kernels`) and Rust reaches it through a plain `extern "C"` block at
   `vendor/libinfer/src/lib.rs:412` and `:486`, wrapped by `src/tsdf_gpu.rs` (1,726 lines).
   So there is **no FFI coupling to break**. Answers open question 3 of the scope doc.

3. **There is no reduction chain in the TSDF.** The scope doc lists one as in-scope. It
   does not exist. `tsdf_hash.cu` (2,178 lines, 10 kernels) contains **zero `__shfl`** and
   exactly one `__shared__` (an evict flag at line 708). Accumulation is per-voxel
   `atomicAdd`; compaction is an `atomicAdd` on an output counter. The warp-level reduction
   work lives in `sim3_icp.cu`, which the scope doc correctly excludes. Consequence: the
   Triton arm's target is an **atomics-heavy irregular scatter**, not a tiled reduction,
   which is Triton's least comfortable ground. ~~Treat A5 as the highest-risk arm.~~
   **Superseded by spike S2 (section 1.2): Triton expresses both the regular update and the
   irregular CAS insertion correctly.** The residual issue is a measured control-flow tax,
   not feasibility.

4. **Integrate and hash insertion are fused today.** `tsdf_hash.h` documents the design:
   *"On integrate-time miss, a thread atomicAdd-claims a free block index from a counter,
   then atomicCAS-inserts into the hash table."* The `atomicCAS` is a single site,
   `tsdf_hash.cu:338`, linear-probing on the `bx` field. The scope doc asks whether they
   are fused and states that if so, splitting them is a Phase 1 prerequisite. **They are.
   It is.** See Phase 2a.

5. **Hardware: RESOLVED, not a blocker. Spike S1 PASSED on sm_120 2026-07-29.**
   Both GPUs are sm_120 (RTX 5070 Ti 16 GB, RTX 5060 8 GB, CUDA 13.0.88, persistence on).
   The `cuda-oxide` docs' prerequisites table ("Ampere+ sm_80+", advanced features
   "sm_80 through sm_100a") is **stale**: `crates/cargo-oxide/src/main.rs:830` lists
   `--arch sm_120  Blackwell (RTX 50 series)`, and `scripts/smoketest.sh` has a
   `blackwell-compile` target pinned to `sm_120a`. Measured result in section 1.1.
   The only remaining reason to want an L4 is the paper's second architecture, which is
   now a *nice-to-have*, not a prerequisite. Answers open question 4.

6. **Test data chosen and validated: TartanAir V2.** Answers open question 5. Details and
   the verified format constants are in section 2. Middlebury / InStereo2K / Booster and
   ETH3D two-view are **isolated stereo pairs** and cannot exercise fusion at all: no
   repeated integrate, no hash growth, no eviction, no incremental marching cubes. They
   remain valid for the depth-accuracy table only.

7. **`Fast-FoundationStereo/cpp/` is not ours.** 2,610 lines of C++ TensorRT runner with a
   GWC-volume plugin and a `profile_speed` app, but it arrived via upstream commit
   `a290ba0`, merge of PR #55 from `wenxind-nvidia`. It is NVIDIA-licensed. Therefore: a
   large head start for Phase 5 as an external dependency, **not** P1.1 authorship
   evidence, and **not** vendorable into an MIT repo.

8. **Open3D needs a source build.** The installed 0.19.0 is the Python wheel. Arm A1 needs
   Open3D as a C++ library, which is a from-source CMake build.

9. **Toolchain installed.** `nightly-2026-04-03` (rustc 1.96.0-nightly, `55e86c996`) with
   `rust-src` and `llvm-tools`, which is the `cuda-oxide` pin. Stable remains default, so
   existing Rust builds are untouched. Triton 3.6.0 and Open3D 0.19.0 are present in
   `.venv`; torch is 2.10.0+cu130.

### 0.5 No code reuse from OSN (decided 2026-07-30)

Nothing is copied from the OSN tree. Prior implementations inform algorithm and
API shape only. This removes arm A0, the vendored `tsdf_hash.cu` (3,458 lines),
and with it the single-oracle correctness design.

**This is a net improvement, not just a constraint.** The old design had A3/A4/A5
ported *from* A0 and then certified *against* A0, which is weakly circular: a
shared misunderstanding of the algorithm would pass the gate. It also put 3,458
lines of code we did not write into a repository whose headline claim is
production C++ authorship.

Correctness now triangulates across three independent sources, in decreasing
order of authority:

1. **Analytic.** Synthetic scenes whose surface is known in closed form (plane
   at known depth, sphere of known radius). Independent of every
   implementation, including ours. This is the tier that makes the design
   stronger than a single oracle: no implementation gets to define "correct".
2. **Third-party.** Open3D `VoxelBlockGrid` on identical input, already built
   and acceptance-tested for both CPU and CUDA (section 0.3.8 / S3).
3. **Cross-arm.** All arms must agree. Catches per-arm bugs, cannot catch a
   mistake common to all of them, hence last.

Cost: **A2 (our CPU C++ arm) moves earlier**, because tiers 1 and 3 need at
least one of our arms running before anything else can be gated. A2 is the right
one to lead with: no GPU nondeterminism, simplest to audit, and it is also the
P1.1 evidence.

### 0.4 Revised effort

The scope doc's 5 to 7 weeks assumed three arms, with the Rust arm reused rather than
written. Five arms, a from-scratch CPU C++ implementation, and an alpha-stage Rust GPU
compiler put this at **9 to 12 weeks part-time**. The fallback in section 7 is how that
compresses. Scaling the work down is Petr's call, not something to do silently.

---

## 1. The platform decision: RESOLVED, develop and benchmark locally

**Local sm_120 is the primary platform.** Spike S1 passed natively. The harness no longer
needs to be remote-capable on day one, which removes the main sequencing constraint the
first draft of this roadmap was built around.

The confound to still avoid: never emit PTX at a virtual arch such as `compute_80` and let
the driver JIT to sm_120 while another arm compiles natively. JIT-from-PTX against native
SASS invalidates the comparison. This is now easy to honour because **every arm can target
sm_120 natively**, and the PTX `.target` directive should be asserted per arm in CI.

A cloud L4 (sm_89, Ada) remains worth one pass late in the project purely for the paper's
second-architecture axis, and it is also the production target (`build.rs` defaults to
`sm_89`). It is no longer on the critical path. Deferred to Phase 6.

### 1.1 Spike S1 result (2026-07-29, VERIFIED)

Two kernels written in Rust, mirroring `tsdf_hash.cu`, run on the RTX 5070 Ti:

- `hash_insert`: 512 threads racing for 64 distinct keys at 8x contention, open-addressed
  linear probing with `compare_exchange`, same integer mix and `MAX_PROBE` bound as
  `tsdf_hash.cu:338`. Result: 64 winners, 0 unresolved threads, all duplicates agreed on
  one slot, 64 distinct slots. **PASS.**
- `voxel_accumulate`: SoA `atomicAdd` of weighted position sums, weight and count, matching
  `hash_integrate_kernel`'s accumulator layout. Totals and per-slot counts exact. **PASS.**

Emitted PTX: `.version 8.7`, **`.target sm_120`** (native, not JIT), `.address_size 64`,
containing `atom.acq_rel.gpu.global.cas.b32` and `atom.global.add.f32` / `.u32`. So the
CAS lowers to the same PTX instruction the CUDA original uses.

`compute-sanitizer --tool racecheck`: **0 hazards, 0 errors, 0 warnings.**

Control: the shipped `atomics` example passes all 20 tests on sm_120, including
`atomic_cas_test`.

**Conclusion: the Rust arm (A4) is viable.** (When written, this section passed the
highest-risk title to A5. Spike S2 in section 1.2 then cleared A5 as well, so **no arm
now carries a feasibility risk.** The dominant remaining risks are schedule and the
Open3D C++ build, not whether the arms can be written.)

### 1.2 Spike S2 result (2026-07-29, VERIFIED). Triton CAN do the irregular part.

Finding 0.3.3 predicted Triton would not be able to express the atomicCAS hash insertion.
**That prediction was wrong.** Both parts pass, mirroring S1 test-for-test at identical
contention (512 threads, 64 distinct keys, 8x duplication) on the RTX 5070 Ti:

- **Test A, regular SoA `atomic_add` accumulation: PASS.** Triton's natural ground.
  24 registers, 0 spills, 9.3 us. Emits `atom.global.gpu.acq_rel.add.f32` / `.add.u32`.
- **Test B, irregular `atomic_cas` linear-probe insertion: PASS.** 64 winners, 0 unresolved,
  all duplicates agreed, 64 distinct slots, scratch slot intact. 38 registers, 0 spills.
  Emits `atom.global.acq_rel.gpu.cas.b32`, which is **the same PTX instruction family** as
  cuda-oxide's `atom.acq_rel.gpu.global.cas.b32` and the CUDA original. PTX `.target sm_120a`.

So the correct finding is not an expressiveness wall. It is a **quantified control-flow tax**:

1. **`tl.atomic_cas` has no `mask` parameter** although `tl.atomic_add` does. Lanes that
   have already resolved cannot be masked off, so they must be redirected to a scratch slot
   holding a sentinel that can never satisfy the CAS. Workable, but it costs a table slot
   and makes the kernel harder to reason about.
2. **No per-lane early exit.** The probe loop compiles to a rolled loop (PTX `bra` count 1,
   438 PTX lines regardless of bound) that always runs the full trip count. The CUDA and
   cuda-oxide versions `break` per thread on success. Measured cost:

   | MAX_PROBE | latency | note |
   |---|---|---|
   | 4 | 7.9 us | |
   | 8 | 12.3 us | |
   | 64 | 65.5 us | same bound as the CUDA original |

   Latency is linear in the bound, an **8.3x penalty from 4 to 64**, even though nearly
   every lane resolves within one or two probes. Triton pays worst case; CUDA pays actual.

That is a better paper result than "cannot express it": a measured penalty with an
identified mechanism, useful to anyone evaluating Triton for volumetric fusion or any
open-addressed GPU hash table.

Gotcha worth recording: `tt.atomic_cas` verifies the cmp operand dtype against the pointee
dtype exactly. Passing a Python `constexpr` int fails to compile with
`'tt.atomic_cas' op failed to verify that ptr type matches cmp type`. Materialise the
sentinel as an explicit `tl.full((BLOCK,), X, tl.int32)`.

**Consequence for A5's scope:** the Phase 2a split is no longer *forced* by Triton's
limitations, though it is still worth doing so all arms share one allocation path and the
comparison measures the update rather than allocation-order luck. It also unlocks a bonus
data point: run A5 both ways, insertion-shared and Triton-does-everything, and report the
tax. Expect Triton to lose the insertion comparison; that loss is the result.

### 1.3 Toolchain setup, reproduce on any fresh machine

`sudo` requires a password on nexxus, so the apt route in the `cuda-oxide` docs is unusable.
What actually worked, entirely in `$HOME`:

1. Ubuntu 24.10 ships LLVM 19 max; `cuda-oxide` needs 21+. Install the official prebuilt
   tarball `LLVM-21.1.8-Linux-X64.tar.xz` into `$HOME/.local/opt/llvm-21`.
   **LLVM 21 is needed only for the clang resource dir** (`lib/clang/21`); `llc` comes from
   the nightly's own bundled LLVM 22.1.2-rust. So the real dependency is the equivalent of
   `libclang-21-dev`, not a full toolchain. Lighter than the docs imply.
2. `rustup toolchain install nightly-2026-04-03` plus components
   `rust-src rustc-dev rust-analyzer clippy llvm-tools`. The pin is in the repo's
   `rust-toolchain.toml`. `rustc-dev` is easy to miss and required.
3. `cargo +nightly-2026-04-03 install --path crates/cargo-oxide`, then `cargo oxide setup`.
4. Env: `PATH` prepends `$HOME/.local/opt/llvm-21/bin` and `/usr/local/cuda/bin`;
   `LLVM_CONFIG_PATH` and `LIBCLANG_PATH` point at the LLVM 21 tree.
5. `cargo oxide doctor` verifies the lot. It also reports the GPU, and
   `compute-sanitizer` 2025.3.1.0 is present, which Phase 1 should use for race checking
   every arm rather than relying on output comparison alone.

---

## 2. Dataset: TartanAir V2 (VERIFIED)

Unreal Engine 4 environments captured through the AirSim plugin. Depth is read from the
renderer's **depth buffer**, so it is exact, dense, hole-free and perfectly registered.
BSD-3-Clause per `tartanair_tools` (one secondary source said MIT; confirm before the paper).

Downloaded sample: `RetroOffice / Data_easy`, `image` + `depth`, `lcam_front` + `rcam_front`,
6 trajectories, 889 stereo frames, **1.9 GB** (1.2 GB RGB, 667 MB depth).
P000 is 129 frames over a 14.07 m trajectory: enough motion to force block allocation and
eviction, small enough to iterate on in seconds.

### 2.1 Format constants, measured not documented

The V1 docs are wrong about V2 in ways that corrupt results **silently** rather than crash.
Hard-code these and assert them on load:

| Property | Correct value | What the docs say |
|---|---|---|
| Resolution | **640 x 640** | 640 x 480 |
| cx, cy | 320, **320** | 320, 240 |
| fx, fy | 320, 320 | correct |
| Baseline | **0.250000 m** (std 1.3e-7 over 129 frames) | 0.25 m, correct |
| Depth encoding | **float32 packed in 4-channel PNG** | 16-bit NPY |
| Depth read | `cv2.imread(p, cv2.IMREAD_UNCHANGED).view("<f4")` | n/a |
| Depth convention | **planar Z**, not ray length | unstated |
| Poses | `tx ty tz qx qy qz qw`, NED (x fwd, y right, z down) | correct |
| `pose` as a modality | **invalid.** Ships with each trajectory | implies valid |

Two traps worth restating:

- **Reading depth with PIL scrambles it.** PIL reorders to RGBA while the on-disk layout is
  BGRA, so the four bytes of each float are permuted. This produces values in a *plausible*
  0.13 to 8 m range with the structure destroyed. A range assertion passes it. Only a
  visual or gradient check catches it.
- **Do not apply `depth_to_dist()` before disparity work.** Stored depth is planar Z, which
  is what disparity is defined against. The package's `depth_to_dist` (`reader.py:128`)
  converts to ray length assuming 90 degree FOV on both axes; applying it here injects a
  radially-varying error that looks like lens distortion.

### 2.2 Rectification: VERIFIED two ways

Geometric, from the pose files over 129 frames:

| Quantity | Measured | Ideal |
|---|---|---|
| Relative rotation L to R | 2.7e-06 deg | 0 |
| Baseline `t_x` (camera right) | +0.250000011 m | 0.25 |
| `t_y` (vertical) | 1.6e-07 m | 0 |
| `t_z` (forward) | 3.5e-07 m | 0 |

Photometric, sampling left at `u + fx*B/Z_right` to reconstruct right, scanning an
artificial vertical shift: error minimises at `dv = 0` with a clean V-shape
(frame 0: 9.80 / 8.79 / **7.78** / 8.96 / 10.08 at dv = -2/-1/0/+1/+2).

Consequences: **feed pairs to FoundationStereo directly, no rectification step.** No
resampling blur, no residual calibration error, no invalid borders to crop, full 640x640
valid, and `Q` reduces to `Z = 80/d`. The warp only closes if `fx*B = 80`, so this
independently cross-validates the intrinsics against docs that were wrong twice.

### 2.3 Two measured constraints on the metric

**Max-disparity ceiling.** Median disparity is 44.5 px but p99 is 289 px and max is 428 px.

| GT disparity | Fraction of pixels |
|---|---|
| > 192 px | 3.18% |
| > 256 px | 1.63% |
| > 416 px | 0.01% |

If the model is exported at max-disp 192, that 3.18% is **structurally unpredictable**, so
those pixels are guaranteed maximal errors concentrated on near-field surfaces, exactly
where integration is densest. Check the configured max disparity of both models, then mask
beyond it and **report the masked fraction**. Including them silently invalidates the table.

**Depth error is quadratic in range**: `dZ = Z^2/(fx*B)`. At this baseline, 1 px of
disparity error costs 0.31 cm at 0.5 m, 1.25 cm at 1 m, 5.0 cm at 2 m, 11.3 cm at 3 m,
**31.3 cm at 5 m**, 80.0 cm at 8 m. Which gives the voxel-sizing constraint:

| Voxel | 1 px error exceeds one voxel beyond |
|---|---|
| 1 cm | 0.89 m |
| 2 cm | 1.26 m |
| 5 cm | 2.00 m |
| 10 cm | 2.83 m |

p99 depth in this scene is 4.78 m, so most of it sits where **depth error dominates voxel
resolution**. This is a result, not a problem: it sharpens the crossover claim into
"backend governs throughput, depth model governs geometric accuracy, and past some range
finer voxels buy nothing". It also lets the existing `tsdf_hash_add_points_radius` camera
gate be set from measurement rather than intuition.

### 2.4 The experiment simulation makes possible

Because exact GT depth and estimated FS depth exist for the same frames, run every arm twice:

- **GT depth into TSDF** = fusion error alone, the oracle upper bound on mesh quality
- **FS depth into TSDF** = fusion error plus depth error, the realistic pipeline

The gap quantifies how much a better depth model buys you at each range. Stronger than
either number alone, and it directly supports the crossover argument.

Honest limitation to state in the paper: FoundationStereo trained on ~1M synthetic pairs,
so scoring it on synthetic data flatters it and measures nothing about sim-to-real. The
saving distinction is that **latency transfers from simulation and accuracy does not.**
Every timing number, which is the point of the paper, is unaffected. Only the depth-accuracy
table needs real data, which is what ETH3D low-res many-view provides (real, laser-scanned
GT mesh, 165 to 352 images per scene).

---

## 3. Order of operations

Sequencing principle: **kill platform risk before building anything, then deliver the C++
gate before the research-flavoured arms.** P1.1 is the gate actually costing interviews, so
it ships before A4 and A5, both of which are more interesting and less urgent.

### Phase 0. De-risking spikes. ~1 week. Do not skip.

Three throwaway spikes. Each answers a question that would force a rewrite if discovered later.

- **S1. `cuda-oxide` on sm_120. DONE 2026-07-29, PASSED.** See section 1.1. Spike source
  kept at `cuda-oxide/crates/rustc-codegen-cuda/examples/tsdf_hash_spike/`; port it into
  the project repo as the seed for A4 rather than rewriting it.
- **S2. Triton hash-TSDF expressiveness. DONE 2026-07-29, PASSED BOTH PARTS.** See
  section 1.2. The prediction that Triton could not express the irregular hash insertion
  was **wrong**; it can, correctly, at a measurable cost. Spike source at
  `scratchpad/triton_spike/s2_triton_spike.py`.
- **S3. Open3D C++ from source.** Long compile, entirely unattended, no risk. Start it first
  and let it run behind S1 and S2. Confirms A1 is buildable and gives the mesh-comparison
  library that Phase 1 needs.

**Gate:** primary benchmark platform chosen and written down; A5's scope fixed to what
Triton demonstrably supports.

### Phase 1. Data, correctness and timing harness. ~1.5 weeks.

Built before any porting, so every later change is measured. Remote-capable from the start
per section 1.

- TartanAir V2 loader with the section 2.1 constants asserted on load, including a
  gradient sanity check on depth so a byte-order regression cannot pass silently.
- Backprojection: planar Z to world points, NED handled once, sky/far-field guard on the
  camera-radius gate.
- Golden reference: thin wrapper over `vendor/libinfer`'s existing C ABI. No reimplementation,
  so it is bit-comparable rather than merely similar.
- Correctness gate: Hausdorff plus mean surface-to-surface via Open3D, tolerance stated.
- Timing harness: p50/p95/p99 per stage, warmup discarded and repeat counts recorded, GPU
  clock and thermal state logged every run. Fixed clocks via `nvidia-smi -lgc` where root
  allows. Without this, Blackwell throttling becomes "the finding".
- Stage attribution: depth inference, disparity to depth, backproject, integrate, surface
  extraction, host transfer.
- CMake skeleton, MIT LICENSE, CI (build plus CPU tests; GPU tests gated/self-hosted).
- Raw CSV output from the first run onward.

**Gate:** the golden reference passes its own correctness gate, and two runs of the same
config agree inside stated noise. If timing is not reproducible yet, no arm is comparable.

### Phase 2. The C++ subsystem. P1.1. ~2.5 weeks. **Highest priority.**

- **2a. Split insertion from integrate (prerequisite).** Per finding 0.3.4 they are fused.
  Extract allocation into a separate pass so all arms share one allocation path. The
  original reason (A5 cannot express insertion) is void per section 1.2. Two reasons now
  stand in its place, and the first makes it **required again**:
    1. **A5a is defined by it.** The shared-insertion Triton variant cannot exist without
       an allocation pass that is callable independently of integrate.
    2. A shared allocator makes the comparison measure the *update* rather than
       allocation-order luck.
  Validate against the golden reference before going on.
- **2b. `include/tsdf/`.** `types.hpp` (block layout, `HashEntry`, accumulator, params),
  `kernel_backend.hpp` (`integrate()`, `extract_surface()`, `extract_mesh()`).
- **2c. `src/volume.cpp`.** RAII device-memory ownership, pool and free-stack management,
  LRU eviction policy, error handling. **This is the centre of gravity for P1.1** and it is
  real work: today that logic is spread through 2,178 lines of `.cu` host code with manual
  `cudaMalloc`/`cudaFree`.
- **2d. A2, the CPU C++ arm.** From scratch, threaded plus SIMD. The strongest pure-C++
  artifact, because no CUDA hides behind it. Also supplies the CPU-vs-GPU axis.
- **2e. A3, the CUDA C++ arm.** Kernels re-authored against the new types.
- **2f. A1, Open3D `VoxelBlockGrid`** behind the same interface. Both its CPU and CUDA
  tensor backends, which anchors A2 and A3 respectively.
- Unit tests per component; all arms pass the Phase 1 correctness gate.

**Gate:** P1.1 evidence is complete and shippable. Repo is presentable to an interviewer
**here**, before the research arms exist. This is the point of no return worth reaching.

### Phase 3. A4, the Rust CUDA arm. ~1.5 weeks (was 2; S1 de-risked it).

`cuda-oxide`, seeded from the S1 spike, which already demonstrates both required
primitives lowering to the expected PTX. The comparison NVIDIA has not published: their own
`cuda-oxide` vs CUDA C++ appendix is an empty placeholder, so there is no public
Rust-to-PTX-vs-nvcc measurement. Report PTX/SASS diffs, register counts and occupancy
alongside wall time, since "why" matters more than "which won". Budget generously: alpha-stage
compiler, expect toolchain spelunking.

### Phase 4. A5, the Triton arm. P1.2. ~1.5 weeks.

**Launched from Rust via AOT cubin, not embedded Python.** Triton exposes the compiled
cubin (`kernel.asm["cubin"]`, verified: 18,808 bytes for the S2 hash-insert kernel, with
metadata `name`, `num_warps=4`, `shared=2048`, `global_scratch_size`). So: compile the
kernels ahead of time in Python as a build step, emit `.cubin` plus a manifest, then load
and launch from the Rust driver with `cuModuleLoadData` / `cuModuleGetFunction` /
`cuLaunchKernel`. **Python becomes a build-time dependency only.**

Why this matters for the paper: every arm then runs under one driver and the benchmark
measures kernel time rather than language runtime. That removes the confound which makes
most published "Triton vs CUDA" comparisons unreliable, and it should be stated explicitly.

RISK to retire early, same treatment as S1/S2: Triton's **launch ABI**. Constexprs are
compiled in rather than passed, argument packing order must match exactly, and Triton 3.x
kernels may require a `global_scratch` pointer argument. Getting it subtly wrong yields
plausible-but-wrong results rather than a crash, so validate the Rust-launched kernel
against the known-good numbers from the Python-launched S2 spike before trusting any
timing.

Seeded from the S2 spike, which already has both kernels working, and from the validated
Rust launch path (`crates/triton-aot`, see docs/TRITON-ABI.md).

**Both variants ship (decided 2026-07-30):**
  * **A5a, insertion shared with A3.** The clean update-only comparison. Depends on the
    Phase 2a split.
  * **A5b, Triton does insertion too.** Reuses the S2 `hash_insert_kernel` directly.

Report `A5b - A5a` as the measured control-flow tax. That is a stronger result than either
number alone, and stronger than the microbenchmark in section 1.2, because it is priced on
the real workload rather than a synthetic contention test.

The claim to report is no longer "cannot express it" but the measured version:
*"Triton matches hand-tuned CUDA on the regular per-block update, and expresses the
irregular open-addressed insertion correctly, but pays a control-flow tax because
`atomic_cas` takes no mask and there is no per-lane early exit, so probe cost is worst-case
rather than actual: 8.3x from MAX_PROBE 4 to 64 on a workload where nearly every lane
resolves in one or two probes."* Sharper and more useful than a feasibility shrug.

Phase 4 is now lower-risk than when planned; if schedule pressure appears, take it from
Phase 3 or 5, not here.

### Phase 5. Depth stage. ~1 week.

**Architecture decided 2026-07-30: Rust libinfer driving pre-built sm_120 TensorRT
engines.** Not the NVIDIA C++ runner (finding 0.3.7: it is upstream-licensed and not
authorship evidence), and not torch.compile.

FoundationStereo (accuracy) and Fast-FoundationStereo (speed) at 576x960 and 320x736,
matching the published deployable ONNX profiles.

Two facts that shape this:

- **libinfer does not build engines** (zero `IBuilder`/`IParser` in `engine.cpp`); it
  loads pre-built `.engine` files. Build them with the existing OSN recipe,
  `scripts/build_blackwell_s120_engines.sh`: `trtexec --fp16
  --builderOptimizationLevel=3 --memPoolSize=workspace:N` with explicit
  min/opt/maxShapes. The two fixed resolutions map to either two engines or two
  optimisation profiles; libinfer supports both via `get_num_profiles` /
  `set_active_profile_async`.
- **libinfer has `infer_device_io`**, device pointers in and out. Depth therefore never
  round-trips to host and feeds `tsdf_hash_add_points_chunk_device` directly. This makes
  the host-transfer ablation (expected to be the paper's second finding) a first-class
  part of the design rather than an afterthought. Re-uses the FP16-safe-set,
custom-operator and Myelin-workaround material. Resolve the max-disparity ceiling from
section 2.3 here.

### Phase 6. Measurement, plots, paper. ~1.5 weeks.

Full matrix: 2 depth arms x 5 fusion arms x 2 resolutions x 2 architectures (L4 sm_89,
local sm_120) x {GT depth, FS depth}. Ablate host-device transfer and stream overlap
explicitly; that is likely the second finding, and it is on-brand given the 300%
throughput result came partly from removing CPU/GPU roundtrips. Publish harness and raw
CSVs. arXiv cs.CV first, then a CVPR/ICCV/ECCV workshop (Embedded Vision, Efficient Deep
Learning for CV).

**Total: 9 to 12 weeks part-time.**

---

## 4. The paper's claim, REFRAMED 2026-07-30

**Superseded by docs/PAPER-FRAMING.md in the project repo.** The crossover
claim below is dead: measured across an 11.8x depth range (Fast-FoundationStereo
13.4 ms, FoundationStereo 157.2 ms), fusion is 0.18% to 2.0% of the pipeline
and never becomes the bottleneck. The paper is now the language comparison,
with the budget share as a secondary negative result.

The original framing is kept below for the record.

## 4b. The paper's claim, as originally revised

The scope doc's crossover claim stands, but a stronger headline emerged: **NVIDIA has
published no performance comparison of `cuda-oxide` against CUDA C++.** Their own
comparison appendix is a placeholder marked "under construction". A rigorous
Rust-to-PTX-vs-nvcc measurement on a real irregular workload is novel in a way that three
bars on a chart is not.

Suggested framing, two findings rather than one:

1. **Crossover.** With a heavyweight depth model in front, the depth stage dominates and
   the fusion backend is irrelevant. With a real-time depth model, fusion becomes the
   bottleneck and backend choice decides whether the pipeline hits frame rate.
2. **Language and codegen.** What Rust-to-PTX and Triton cost against hand-tuned CUDA on
   an atomics-heavy irregular workload, with the boundary of what each can express.

Section 2.3's depth-error curve supports a third, quieter point: past a measurable range,
finer voxels buy nothing, because depth error exceeds voxel size.

---

## 5. Methodology commitments

Non-negotiable, and the actual differentiator given five years of statistical accept/reject work:

- Correctness gate precedes all timing. A faster backend that drifts is not a result.
- Distributions, not means. p50/p95/p99, warmup discarded, repeat counts stated.
- Fixed clocks, thermal state logged.
- No JIT-vs-SASS comparisons across arms. See section 1.
- Report every masked or dropped pixel fraction, including the max-disparity ceiling.
- Publish harness plus raw CSVs.

---

## 6. Open items

- ~~S1, `cuda-oxide` on sm_120~~. **DONE, PASSED 2026-07-29.** See section 1.1.
- ~~S2, Triton hash-TSDF feasibility~~. **DONE, PASSED BOTH PARTS 2026-07-29.** See
  section 1.2. A5 is no longer the highest-risk arm; no arm now carries feasibility risk.
- ~~Confirm Triton's role is fusion~~. **RESOLVED 2026-07-30: fusion. No blockers remain.**
- **S3, Open3D C++ from source: NOT YET RUN.** Now the only unstarted Phase 0 spike, and
  by elimination the largest remaining unknown.
- Cloud L4: deferred to Phase 6, second-architecture axis only. Not on the critical path.
- Root for `nvidia-smi -lgc` fixed clocks.
- TartanAir license: BSD-3-Clause per repo, MIT per one secondary source.
- Configured max disparity for both FoundationStereo variants.
- FoundationStereo research repo **code** license, distinct from the NVIDIA Open Model
  License on the weights, before anything beyond benchmarking.
- Dense vs hash scope: recommend **hash only** for the ported arms. The 784-line dense
  sliding-window grid adds a third code path across five arms for no gain to the claim,
  since the crossover argument is about hash fusion. It stays in the reference as context.
- Environment spread: 1 env is 1.9 GB at image+depth on the front stereo pair, so a
  10-env spread is roughly 40 GB and 30 envs about 120 GB. Full V2 across all modalities
  and all 12 cameras extrapolates to multiple TB; never needed.

---

## 7. Fallback if time compresses

Ordered by what to cut first. **Do not let the paper's ambition delay the C++ evidence.**

1. Cut Phase 5, the depth arms. Benchmark the fusion arms on pre-recorded GT depth.
   Still publishable as a systems note, still fully satisfies P1.1 and P1.2. The crossover
   claim weakens to a fusion-throughput claim.
2. Cut Phase 3, A4/`cuda-oxide`, if S1 shows the toolchain fighting back. Highest-variance
   arm.
3. Cut A1, Open3D, if the source build fights back. Costs the P1.5 side benefit only.
4. **Never cut Phase 2.** It is the gate actually costing interviews. If only one phase
   ever ships, it is this one.
