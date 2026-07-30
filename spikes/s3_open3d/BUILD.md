# Building Open3D 0.19 as a C++ library on CUDA 13

## Verdict

- **CPU-only: WORKS.** Clean build, 0 errors. `VoxelBlockGrid` and the mesh
  comparison API both verified from C++ (see `acceptance/`).
- **CUDA module: BLOCKED.** Open3D 0.19 predates CUDA 13 and its own GPU
  hashmap does not compile against Thrust 3.

## CPU-only (recommended)

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA_MODULE=OFF \
  -DBUILD_PYTHON_MODULE=OFF -DBUILD_GUI=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_UNIT_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_WEBRTC=OFF \
  -DWITH_OPENMP=ON -DCMAKE_INSTALL_PREFIX=$HOME/.local/opt/open3d-cpu
make -j12 && make install
```

Consume with `find_package(Open3D REQUIRED)` and
`-DOpen3D_ROOT=$HOME/.local/opt/open3d-cpu`.

## CUDA attempt, for the record

Configure succeeds with `-DCMAKE_CUDA_ARCHITECTURES=120`. Open3D's own defaults
stop at sm_90, but it honours a user-provided value and nvcc 13 accepts sm_120.
The build then hits four CUDA 13 incompatibilities, each silent until the
previous is cleared.

Three are in `stdgpu`, which Open3D pins to a 2023 commit, and are fixed by
`stdgpu.cmake.patched` plus `patch_cuda13.sh`:

1. `thrust/version.h` moved to `include/cccl/thrust/`. The finder's regex
   returns empty and reports `Found unsuitable version "ERROR.ERROR.ERROR"`.
2. `LIBCUDACXX_INCLUDE_DIR` and `CUB_INCLUDE_DIR` need the same CCCL root.
3. `cudaDeviceProp::clockRate` was removed; stdgpu uses it only to print device
   info, so it becomes `cudaDeviceGetAttribute(cudaDevAttrClockRate)`.

Apply fix 3 as a `PATCH_COMMAND`, not an edit to the extracted tree:
`ExternalProject` re-extracts sources, so an in-tree edit vanishes silently on a
clean rebuild. The script is idempotent.

The fourth is not fixable in a patch. `cpp/open3d/core/hashmap/CUDA/
CreateCUDAHashBackend.cu` fails against Thrust 3.0.1 with `initial value of
reference to non-const must be an lvalue`, `reference to void is not allowed`,
and a `thrust::pointer` conversion failure. This is Thrust 3 API churn in
`device_vector` iterator and pointer semantics, in Open3D core.

## Acceptance test

`acceptance/` builds against the installed library and checks the two things the
project actually needs:

```
Open3D 0.19.0
  unique block coords: 3700
  hashmap size after integrate: 3700
  mesh: 463761 vertices, 924800 triangles
  sphere r=1.00 vs r=1.02: mean surface dist 0.0460, hausdorff 0.1216
VoxelBlockGrid (A1)        : PASS
Mesh comparison (P1 gate)  : PASS
```

Note `ExtractTriangleMesh` defaults to `weight_threshold = 3.0`, so a single
integration extracts nothing. Pass a lower threshold or integrate repeatedly.
This looks exactly like a broken TSDF and is not one.

The mean surface distance of 0.046 between spheres of radius 1.00 and 1.02 is
dominated by point-sampling spacing (2000 points on a unit sphere sit ~0.09
apart), not by the 0.02 radius difference. Use dense correspondences for the
real gate.
