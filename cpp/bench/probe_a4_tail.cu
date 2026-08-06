// Why does arm A4's allocate stage have a heavy tail?
//
// Interleaved timing showed A4 allocate at p50 0.052 ms but p95 0.187 and p99
// 0.235, roughly 4x, where A3 stays flat at 0.051. A tail that size matters for
// a frame-rate claim in a way a median does not, so it needs attributing
// before the number is quoted.
//
// Candidate mechanisms:
//   H1  Device-side. The spin on block_idx publication occasionally waits a
//       long time. Would show up in the CUDA-event time.
//   H2  Host-side. A4's launch path allocates an Arc per call in
//       default_stream() and wraps five caller-owned pointers in DeviceBuffer.
//       Would show up in wall time but NOT in the event time.
//   H3  Periodic external. Driver or allocator behaviour on some cadence.
//       Would show up as outliers at regular iteration indices.
//
// Records both clocks per iteration and prints the sequence, so the shape of
// the distribution decides rather than a summary statistic.

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "analytic_scenes.hpp"
#include "osn_tsdf/c_api.h"

extern "C" {
int32_t a4_init();
int32_t a4_allocate_blocks(const OsnTsdfDeviceView*, uint64_t, int32_t, float, float, float, float);
}

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    if (a4_init() != 0) { std::printf("a4_init failed\n"); return 1; }

    const auto pts = scenes::sphere(0.5f, 400, 800);
    const int32_t n = (int32_t)(pts.size() / 3);
    float* d = nullptr;
    cudaMalloc(&d, pts.size() * sizeof(float));
    cudaMemcpy(d, pts.data(), pts.size() * sizeof(float), cudaMemcpyHostToDevice);

    OsnTsdfCudaVolume* v = osn_tsdf_cuda_create(0.01f, 1 << 14, -1.0f, 32.0f);
    OsnTsdfDeviceView dv{};
    osn_tsdf_cuda_device_view(v, &dv);

    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);

    std::vector<double> gpu_ms(iters), host_ms(iters);
    for (int i = 0; i < iters; ++i) {
        osn_tsdf_cuda_reset(v);
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        cudaEventRecord(a);
        a4_allocate_blocks(&dv, (uint64_t)d, n, 0, 0, 0, 0);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        host_ms[i] = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0).count();
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        gpu_ms[i] = ms;
    }

    std::printf("A4 allocate, %d iterations. gpu = CUDA events, host = wall around the call.\n\n",
                iters);
    std::printf("  first 24 iterations (gpu / host, ms):\n   ");
    for (int i = 0; i < std::min(24, iters); ++i)
        std::printf(" %.3f/%.3f", gpu_ms[i], host_ms[i]);
    std::printf("\n\n");

    // Which iterations are the outliers, and do they line up in both clocks?
    std::vector<int> idx(iters);
    for (int i = 0; i < iters; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int x, int y) { return gpu_ms[x] > gpu_ms[y]; });
    std::printf("  10 slowest by GPU time:\n");
    for (int k = 0; k < std::min(10, iters); ++k)
        std::printf("    iter %4d   gpu %.3f   host %.3f\n", idx[k], gpu_ms[idx[k]],
                    host_ms[idx[k]]);

    auto pct = [](std::vector<double> v, double p) {
        std::sort(v.begin(), v.end());
        return v[std::min(v.size() - 1, (size_t)(v.size() * p))];
    };
    std::printf("\n  gpu  p50 %.3f  p95 %.3f  p99 %.3f  max %.3f\n", pct(gpu_ms, 0.5),
                pct(gpu_ms, 0.95), pct(gpu_ms, 0.99), pct(gpu_ms, 1.0));
    std::printf("  host p50 %.3f  p95 %.3f  p99 %.3f  max %.3f\n", pct(host_ms, 0.5),
                pct(host_ms, 0.95), pct(host_ms, 0.99), pct(host_ms, 1.0));
    std::printf("\n  host-minus-gpu p50 %.3f ms (launch path overhead)\n",
                pct(host_ms, 0.5) - pct(gpu_ms, 0.5));

    // Is the tail in the launch path, or in cudaEventSynchronize? Re-measure
    // with no events at all: wall clock around the call, then an explicit
    // device sync outside the timed region.
    std::vector<double> call_ms(iters), sync_ms(iters);
    for (int i = 0; i < iters; ++i) {
        osn_tsdf_cuda_reset(v);
        cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        a4_allocate_blocks(&dv, (uint64_t)d, n, 0, 0, 0, 0);
        call_ms[i] = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0).count();
        t0 = std::chrono::steady_clock::now();
        cudaDeviceSynchronize();
        sync_ms[i] = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0).count();
    }
    std::printf("\n  no-event decomposition:\n");
    std::printf("    launch call only   p50 %.3f  p95 %.3f  p99 %.3f  max %.3f\n",
                pct(call_ms, 0.5), pct(call_ms, 0.95), pct(call_ms, 0.99), pct(call_ms, 1.0));
    std::printf("    device sync after  p50 %.3f  p95 %.3f  p99 %.3f  max %.3f\n",
                pct(sync_ms, 0.5), pct(sync_ms, 0.95), pct(sync_ms, 0.99), pct(sync_ms, 1.0));

    osn_tsdf_cuda_destroy(v);
    cudaFree(d);
    return 0;
}
