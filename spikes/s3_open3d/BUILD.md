# Building Open3D 0.19 as a C++ library on CUDA 13

## Verdict: WORKS, CPU and CUDA both

Requires backporting **two files from Open3D `main`** onto the v0.19.0 tree.
No Open3D source changes are needed.

```
Open3D 0.19.0
CUDA available: yes

--- VoxelBlockGrid (A1 arm) ---
  CPU:0   blocks  3700   vertices  463761   triangles  924800
  CUDA:0  blocks  3700   vertices  463761   triangles  924800

--- Mesh comparison (Phase 1 correctness gate) ---
  r=1.00 vs r=1.01 : mean 0.0166  hausdorff 0.0465
  r=1.00 vs r=1.10 : mean 0.1010  hausdorff 0.1094

VoxelBlockGrid CPU                 PASS
VoxelBlockGrid CUDA                PASS
CPU/CUDA block-count agreement     PASS
Mesh comparison (P1 gate)          PASS
```

CPU and CUDA agree exactly on block count, vertex count and triangle count,
which is what A1 needs to be usable as a baseline on either device.

## The fix: backport upstream PR #7398

[PR #7398](https://github.com/isl-org/Open3D/pull/7398), "Upgrade stdgpu to fix
compilation errors on Jetson Thor with Ubuntu 24.04 and Cuda 13.0", merged
2026-01-15. It touches only `3rdparty/stdgpu/`, so it backports cleanly onto
v0.19.0 without touching Open3D source.

```bash
cd <open3d-src>
B=https://raw.githubusercontent.com/isl-org/Open3D/main
curl -sL -o 3rdparty/stdgpu/stdgpu.cmake $B/3rdparty/stdgpu/stdgpu.cmake
curl -sL -o 3rdparty/stdgpu/fix-thrust-is-proxy-reference.patch \
    $B/3rdparty/stdgpu/fix-thrust-is-proxy-reference.patch
rm -rf build/stdgpu     # PATCH_COMMAND runs git init + git apply; needs a clean tree
cd build && cmake . && make -j12
```

Copies of both files are here as `stdgpu.cmake.upstream` and
`fix-thrust-is-proxy-reference.patch`.

What the backport does:

1. **Bumps stdgpu** from `2588168d` (2023) to `d7c07d05`. The newer version
   defines `device_ptr<T>` as `thrust::pointer<T, thrust::device_system_tag>`
   directly, so the hand-rolled `std::iterator_traits<stdgpu::device_ptr<T>>`
   specialisation that hard-errored on `T = void` is gone. It also already uses
   `cudaDeviceGetAttribute(cudaDevAttrClockRate, ...)` instead of the
   `cudaDeviceProp::clockRate` field CUDA 13 removed.
2. **Points Thrust at CCCL** when `CUDAToolkit_VERSION >= 13.0`, since CUDA 13
   moved Thrust, CUB and libcu++ under `include/cccl/`.
3. **Applies `fix-thrust-is-proxy-reference.patch`**, renaming stdgpu's Thrust
   trait specialisations from `is_proxy_reference` to `is_wrapped_reference`.

## Why item 3 matters more than it looks

The build originally also failed in Open3D's **own** code, at
`cpp/open3d/core/hashmap/CUDA/SlabNodeManager.h:268`:

```cpp
thrust::device_vector<uint32_t> slabs_per_superblock(kSuperBlocks);
std::vector<int> result(num_super_blocks);
thrust::copy(slabs_per_superblock.begin(), slabs_per_superblock.end(),
             result.begin());          // uint32_t (device) -> int (host)
```

with `no suitable constructor exists to convert from "unsigned int *" to
thrust::pointer<...>` inside `contiguous_storage`. That looked like a second,
independent root cause needing an Open3D patch, and `SlabNodeManager.h` is
**unchanged on main**, which seemed to contradict upstream building on CUDA 13.

It resolved itself. `is_wrapped_reference` is exactly the trait Thrust's
cross-system copy consults when deciding whether to stage through a temporary
buffer. With stdgpu registering its specialisations under the old name, Thrust 3
saw an unspecialised trait, took the wrong dispatch branch, and failed inside
`contiguous_storage`. Fix the trait name and the copy compiles untouched.

So the two apparently independent failures were one cause with two symptoms.
`CreateCUDAHashBackend.cu` compiles clean after the backport, and no Open3D
source edit is required.

## Configure

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA_MODULE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DBUILD_PYTHON_MODULE=OFF -DBUILD_GUI=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_UNIT_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_WEBRTC=OFF \
  -DWITH_OPENMP=ON -DCMAKE_INSTALL_PREFIX=$HOME/.local/opt/open3d
make -j12 && make install
```

Open3D 0.19's own arch defaults stop at sm_90, but it honours a user-provided
`CMAKE_CUDA_ARCHITECTURES` and nvcc 13 accepts sm_120, so pass it explicitly.

For a CPU-only build use `-DBUILD_CUDA_MODULE=OFF`; the backport is then inert
and unnecessary.

## Consuming it: link the CUDA runtime yourself

A **static** Open3D built with `BUILD_CUDA_MODULE=ON` does not propagate the
CUDA runtime to a plain `LANGUAGES CXX` consumer. Linking fails with undefined
references to `cudaMalloc`, `cudaEventCreate`, `cudaDeviceGetAttribute` and
friends. Add:

```cmake
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(<tgt> PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
endif()
```

See `acceptance/CMakeLists.txt`.

## Three traps in the acceptance test itself

All look like real failures and are not:

- **`ExtractTriangleMesh` defaults to `weight_threshold = 3.0`**, so a single
  integration extracts zero vertices. That looks exactly like a broken TSDF.
  Integrate repeatedly or lower the threshold.
- **Sphere-to-sphere distances are dominated by point-sampling spacing**, not by
  the radius difference: 2000 points on a unit sphere sit ~0.09 apart, which
  swamps a 0.02 radius change. The test therefore asserts **monotonicity**
  across two perturbation sizes rather than absolute values, because
  monotonicity is the property the Phase 1 gate actually depends on.
- **`core::Device::IsAvailable()` is a member function**, not a static taking a
  `Device`.

## Caveat

Upstream CI builds CUDA wheels against **CUDA 12.6.3**; there is no CUDA 13 job
in the matrix. PR #7398 was contributed for Jetson Thor, so CUDA 13 is a
fixed-on-request configuration rather than a covered one. This build works, but
it is not a well-trodden path. If A1 ever becomes load-bearing rather than a
droppable baseline, building against CUDA 12.6 to match upstream CI is the more
conservative option.
