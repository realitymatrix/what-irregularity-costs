// What is actually being measured when the harness times "extract"?
//
// The extraction stage was reported at 1.85 ms on the RTX 5070 Ti and 12.72 ms
// on the RTX 5060, a 6.9x device difference against a 2.33x SM ratio. Nothing
// the integrate stages do looks like that, so before deciding whether
// extraction joins the language comparison (docs/ARM-SCOPE.md) it is worth
// knowing what the number is made of.
//
// `CudaVolume::extract_mesh` does four things inside one call:
//
//   1. cudaMalloc of capacity * 6 floats  (100 MiB at the harness's capacity)
//   2. the marching-tets kernel
//   3. cudaMemcpy device-to-host of the vertices actually produced (20.3 MiB)
//   4. cudaFree
//
// Only (2) is a kernel comparison. (1) and (4) are allocator behaviour and (3)
// is PCIe, and the two cards are not on comparable links: device 0 negotiates
// gen2 x16 while device 1 is on gen1 x2, up to 16x less host bandwidth.
//
// The split is measured by running extraction twice with different
// `min_weight`. At the normal threshold every voxel passes and the full
// readback happens. At an impossible threshold no voxel passes, so the kernel
// still sweeps every block and every allocation still happens, but the large
// memcpy is skipped. The difference is the readback.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "analytic_scenes.hpp"
#include "osn_tsdf/c_api.h"

namespace {

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Timer {
    cudaEvent_t a{}, b{};
    Timer() { cudaEventCreate(&a); cudaEventCreate(&b); }
    void start() { cudaEventRecord(a); }
    double stop_ms() {
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        return ms;
    }
};

}  // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 15;
    const float voxel = 0.01f;

    cudaSetDevice(0);
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    cudaFree(nullptr);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    int width = 0, gen = 0;
    cudaDeviceGetAttribute(&width, cudaDevAttrPciBusId, 0);  // placeholder, see print below
    (void)width;
    (void)gen;

    std::printf("=== what is in the extract measurement? ===\n");
    std::printf("  device: %s | %d SMs\n", prop.name, prop.multiProcessorCount);
    std::printf("  (PCIe link: read from nvidia-smi alongside this run)\n\n");

    const auto pts = scenes::sphere(0.5f, 400, 800);
    const int32_t n = (int32_t)(pts.size() / 3);
    float* d_pts = nullptr;
    cudaMalloc(&d_pts, pts.size() * sizeof(float));
    cudaMemcpy(d_pts, pts.data(), pts.size() * sizeof(float), cudaMemcpyHostToDevice);

    OsnTsdfCudaVolume* v = osn_tsdf_cuda_create(voxel, 1 << 14, -1.0f, 32.0f);
    if (!v) { std::printf("volume allocation failed\n"); return 1; }
    osn_tsdf_cuda_allocate_blocks(v, d_pts, n, 0, 0, 0, 0);
    osn_tsdf_cuda_update_voxels(v, d_pts, n, 0, 0, 0, 0);
    osn_tsdf_cuda_synchronize(v);

    const int32_t cap = 4 << 20;
    std::vector<float> mesh((size_t)cap * 6);

    // Warmup, and learn how many vertices the full path actually transfers.
    const int32_t nv = osn_tsdf_cuda_extract_mesh(v, mesh.data(), nullptr, cap, 0.5f, 0.0f);
    const double xfer_mib = (double)nv * 6 * sizeof(float) / 1048576.0;
    const double alloc_mib = (double)cap * 6 * sizeof(float) / 1048576.0;
    std::printf("  vertices %d  ->  readback %.1f MiB, scratch cudaMalloc %.0f MiB\n\n", nv,
                xfer_mib, alloc_mib);

    Timer t;
    std::vector<double> full, empty;
    for (int i = 0; i < reps + 3; ++i) {
        t.start();
        osn_tsdf_cuda_extract_mesh(v, mesh.data(), nullptr, cap, 0.5f, 0.0f);
        const double f = t.stop_ms();

        // Impossible weight: the kernel runs over every block and every
        // allocation still happens, but no vertex survives, so the large
        // readback is skipped.
        t.start();
        osn_tsdf_cuda_extract_mesh(v, mesh.data(), nullptr, cap, 1e9f, 0.0f);
        const double e = t.stop_ms();

        if (i >= 3) { full.push_back(f); empty.push_back(e); }
    }

    const double mf = median(full), me = median(empty);
    std::printf("  full extract (kernel + alloc + %.1f MiB readback)  %8.3f ms\n", xfer_mib, mf);
    std::printf("  no-readback  (kernel + alloc only)                 %8.3f ms\n", me);
    std::printf("  difference   (the device-to-host transfer)         %8.3f ms  = %.1f%% of full\n",
                mf - me, 100.0 * (mf - me) / mf);
    if (mf > me)
        std::printf("  implied host bandwidth                            %8.2f GiB/s\n",
                    xfer_mib / 1024.0 / ((mf - me) / 1000.0));
    std::printf("\n  Only the kernel part of the no-readback figure is a language-comparable\n");
    std::printf("  quantity, and even that still contains a %.0f MiB cudaMalloc/cudaFree.\n",
                alloc_mib);

    osn_tsdf_cuda_destroy(v);
    cudaFree(d_pts);
    return 0;
}
