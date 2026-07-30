# triton_stereo_depth_inference

Where does the time go in a foundation-model stereo reconstruction pipeline,
and at what point does the fusion backend start to matter?

Foundation-model stereo depth (FoundationStereo / Fast-FoundationStereo) feeding
GPU TSDF fusion, with the fusion stage implemented five ways behind one
interface, measured stage by stage.

**Status: Phase 0 (de-risking spikes) COMPLETE, all three passed.** No project
code yet by design: the roadmap puts spikes before scaffolding, because each one
answers a question that would force a rewrite if discovered later. All three
overturned a prediction in the roadmap, every time in the project's favour.
**No arm carries feasibility risk.** Next is Phase 1: the correctness and timing
harness.

## The claim

The interesting result is a crossover, not a leaderboard. With a heavyweight
depth model in front, the depth stage dominates and the fusion backend is
irrelevant. Swap in a real-time depth model and fusion becomes the bottleneck,
at which point backend choice decides whether the pipeline hits frame rate.

A second claim emerged during Phase 0: **NVIDIA has published no performance
comparison of `cuda-oxide` (their Rust-to-PTX compiler) against CUDA C++.**
Their own comparison appendix is an empty placeholder marked "under
construction". A rigorous measurement on a real irregular workload is novel.

## Fusion arms

| # | Arm | Purpose |
|---|-----|---------|
| A0 | Existing Rust/CUDA hash TSDF | Golden correctness reference, not a language arm |
| A1 | Open3D `VoxelBlockGrid` (C++) | External baseline |
| A2 | Our own C++ TSDF (CPU, C++17/20) | Production C++ authorship evidence |
| A3 | Our own CUDA C++ TSDF | Kernel authorship, the performance reference point |
| A4 | Our own Rust CUDA TSDF via `cuda-oxide` | Rust-to-PTX vs nvcc codegen |
| A5a | Triton fusion, insertion shared with A3 | Clean update-only comparison vs A3 |
| A5b | Triton fusion, insertion in Triton too | Prices the control-flow tax end to end |

A5 ships as two variants because spike S2 showed Triton *can* express the
irregular CAS insertion. `A5b - A5a` is then a direct measurement of Triton's
control-flow tax on the real workload, rather than an inference from a synthetic
contention test.

Triton here means **OpenAI Triton** (`@triton.jit`), the kernel DSL, not NVIDIA
Triton Inference Server. Its role is the fusion backend, not depth
pre/post-processing: rectification and normalisation kernels are memory-bound
and prove nothing.

## Phase 0 results

| Spike | Question | Result |
|---|---|---|
| S1 | `cuda-oxide` on sm_120 with the TSDF primitives | **PASS** |
| S2 | Triton expressing the irregular atomicCAS insertion | **PASS**, with a measured cost |
| S3 | Open3D as a C++ library on CUDA 13 | **PASS**, CPU and CUDA |

Headlines:

- **S1 killed a blocker that never existed.** The `cuda-oxide` docs say sm_100a
  is the ceiling; the source says `--arch sm_120  Blackwell (RTX 50 series)`.
  Local development is viable and cloud GPU left the critical path.
- **S2 refuted the prediction that Triton could not express the hash insert.**
  It can. The real finding is a quantified control-flow tax: `tl.atomic_cas`
  takes no mask and there is no per-lane early exit, so probe cost is
  worst-case rather than actual, an **8.3x penalty** from MAX_PROBE 4 to 64.
- **S3 found Open3D 0.19 is not CUDA 13 clean**, then unblocked it by
  backporting upstream PR #7398 (two files, `3rdparty/stdgpu/` only). CPU and
  CUDA now agree exactly: 3700 blocks, 463761 vertices on both devices. No
  Open3D source changes needed.

Full detail with measurements: [docs/SPIKE-RESULTS.md](docs/SPIKE-RESULTS.md).

## Dataset

TartanAir V2 (Unreal Engine 4 via AirSim, depth from the renderer's depth
buffer, so exact and dense). Chosen because TSDF fusion needs a **sequence with
poses**: Middlebury, InStereo2K, Booster and ETH3D two-view are isolated pairs
and cannot exercise fusion at all.

Ground-truth poses are a feature here, not a compromise. Comparing five
implementations of one algorithm means any mesh difference must be attributable
to the backend; SLAM drift would interact with hash allocation order and
eviction timing and turn the correctness gate into a measure of pose error.

`tools/tartanair.py` encodes format constants measured from the data, because
the V1 docs are wrong about V2 in ways that corrupt results **silently**:
resolution is 640x640 not 640x480 (so cy = 320 not 240), and depth is float32
bit-packed into a 4-channel PNG that must be read with cv2, since PIL permutes
the bytes and yields a plausible range with the structure destroyed.

Rectification is verified two ways rather than assumed: relative rotation
2.7e-06 deg with baseline 0.250000011 m and vertical offset at 1e-07 m, plus a
photometric warp that minimises at zero vertical shift.

Run `python tools/tartanair.py <traj_dir>` on any new environment before
trusting it.

## Layout

```
docs/ROADMAP.md          full plan, phasing, effort, fallbacks
docs/SPIKE-RESULTS.md    Phase 0 measurements
spikes/s1_cuda_oxide/    Rust kernels: CAS hash insert + SoA accumulate
spikes/s2_triton/        the same two kernels in Triton, for comparison
spikes/s3_open3d/        CUDA 13 backport + CPU/CUDA acceptance test
tools/tartanair.py       dataset loader + format verifier
```

## Methodology commitments

Non-negotiable, because most engineering benchmark papers are rejected for
sloppy measurement:

- Correctness gate precedes all timing. A faster backend that drifts is not a
  result.
- Distributions, not means: p50/p95/p99, warmup discarded, repeat counts stated.
- Fixed clocks, thermal state logged, or Blackwell throttling silently becomes
  "the finding".
- No JIT-vs-SASS comparisons across arms. Assert the PTX `.target` per arm in CI.
- Report every masked or dropped pixel fraction, including the max-disparity
  ceiling (3.18% of pixels here exceed 192 px disparity).
- Publish the harness and raw CSVs.

## Environment

Ubuntu 24.10, RTX 5070 Ti + RTX 5060 (both sm_120), CUDA 13.0.88, gcc 14.2,
Rust nightly-2026-04-03, Triton 3.6.0, torch 2.10.0+cu130.

Setup notes for a fresh machine, including the no-root LLVM 21 install, are in
[docs/SPIKE-RESULTS.md](docs/SPIKE-RESULTS.md).

## Licence

MIT for this repository's own code. Third-party components keep their own
licences: `cuda-oxide` and `stdgpu` patches are derived from Apache-2.0 sources,
FoundationStereo weights are under the NVIDIA Open Model License, and TartanAir
is BSD-3-Clause.
