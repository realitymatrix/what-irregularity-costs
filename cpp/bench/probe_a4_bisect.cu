// Bisect arm A4's allocate gap by removing one component at a time.
//
// The gap is 1.24x to 4.06x depending on workload, and every mechanism proposed
// for it has been eliminated: register pressure, `f32::clamp`, `read_volatile`,
// static instruction count, and memory-ordering scope (docs/A3-A4-GAP.md).
//
// Two facts narrow where to look:
//
//   1. It is a FIXED cost, not a multiplier. At 20k points A3 takes 3 us and A4
//      takes 21; at 1.28M points the ratio is only 1.51x. Raising the batch from
//      8 to 64 amortises A3 (0.005 -> 0.003 ms) and leaves A4 pinned at 0.021,
//      which rules out host enqueue, FFI and stream lookup, because all of those
//      would amortise. It is a device-side floor.
//   2. It is specific to the insert path. At the same cell the update kernel is
//      a flat 1.15x with no floor, and update differs from allocate by exactly
//      three things: the compare-exchange, the counter atomicAdd, and the spin
//      that waits for a peer to publish `block_idx`.
//
// So this prices those three. Each variant is a copy of the real kernel with
// one component disabled through a const generic, so the geometry walk and the
// probe loop are identical and the delta is attributable. Every variant is
// DELIBERATELY WRONG and exists only to be timed.
//
// Read the output as: whichever variant recovers A3's time contains the cost.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "analytic_scenes.hpp"
#include "osn_tsdf/c_api.h"

extern "C" {
int32_t a4_init();
int32_t a4_allocate_blocks(const OsnTsdfDeviceView*, uint64_t, int32_t, float, float, float, float);
int32_t a4_allocate_nospin(const OsnTsdfDeviceView*, uint64_t, int32_t, float, float, float, float);
int32_t a4_allocate_nocas(const OsnTsdfDeviceView*, uint64_t, int32_t, float, float, float, float);
int32_t a4_allocate_nocount(const OsnTsdfDeviceView*, uint64_t, int32_t, float, float, float, float);
int32_t a4_allocate_nofence(const OsnTsdfDeviceView*, uint64_t, int32_t, float, float, float, float);
int32_t a4_allocate_nopublish(const OsnTsdfDeviceView*, uint64_t, int32_t, float, float, float, float);
int32_t a4_allocate_cas128(const OsnTsdfDeviceView*, uint64_t, int32_t, float, float, float, float);
int32_t a4_allocate_slotidx(const OsnTsdfDeviceView*, uint64_t, int32_t, float, float, float, float);
}

namespace {

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
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

using AllocFn = int32_t (*)(const OsnTsdfDeviceView*, uint64_t, int32_t, float, float, float,
                            float);

}  // namespace

int main(int argc, char** argv) {
    // Default to the cell where the floor is 7x and therefore unmissable.
    const int n_theta = argc > 1 ? std::atoi(argv[1]) : 100;
    const int n_phi = argc > 2 ? std::atoi(argv[2]) : 200;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 20;
    const int BATCH = argc > 4 ? std::atoi(argv[4]) : 32;
    const float voxel = 0.01f;

    cudaSetDevice(0);
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    cudaFree(nullptr);
    if (a4_init() != 0) { std::printf("A4 unavailable\n"); return 1; }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    const auto pts = scenes::sphere(0.5f, n_theta, n_phi);
    const int32_t n = (int32_t)(pts.size() / 3);

    std::printf("=== A4 allocate bisect ===\n");
    std::printf("  device %s | %d SMs | %d points | batch %d | %d reps\n\n", prop.name,
                prop.multiProcessorCount, n, BATCH, reps);

    float* d_pts = nullptr;
    cudaMalloc(&d_pts, pts.size() * sizeof(float));
    cudaMemcpy(d_pts, pts.data(), pts.size() * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<OsnTsdfCudaVolume*> vols((size_t)BATCH, nullptr);
    std::vector<OsnTsdfDeviceView> dvs((size_t)BATCH);
    for (int j = 0; j < BATCH; ++j) {
        vols[j] = osn_tsdf_cuda_create(voxel, 1 << 14, -1.0f, 32.0f);
        if (!vols[j]) { std::printf("volume alloc failed\n"); return 1; }
        osn_tsdf_cuda_device_view(vols[j], &dvs[j]);
    }

    struct Arm {
        const char* name;
        AllocFn fn;       // null means arm A3, launched through its own entry point
        const char* what;
    } arms[] = {
        {"A3 cuda", nullptr, "reference, full insert"},
        {"A3 -cas", nullptr, "probe and read, never insert"},
        {"A4 full", a4_allocate_blocks, "baseline, correct"},
        {"A4 -spin", a4_allocate_nospin, "no wait for a peer's block_idx"},
        {"A4 -cas", a4_allocate_nocas, "probe and read, never insert"},
        {"A4 -count", a4_allocate_nocount, "no shared counter atomicAdd"},
        {"A4 -fence", a4_allocate_nofence, "no threadfence before publishing idx"},
        {"A4 -publish", a4_allocate_nopublish, "CAS + counter, but never publish (no spin)"},
        {"A4 slotidx", a4_allocate_slotidx, "CORRECT: block index = hash slot"},
    };

    Timer t;
    std::vector<double> samples[9];
    // Interleave, as the main harness does: a per-arm block would let clock ramp
    // masquerade as a difference between variants.
    for (int i = 0; i < reps + 5; ++i) {
        for (int k = 0; k < 9; ++k) {
            const int a = (i + k) % 9;
            for (int j = 0; j < BATCH; ++j) osn_tsdf_cuda_reset(vols[j]);
            cudaDeviceSynchronize();
            t.start();
            for (int j = 0; j < BATCH; ++j) {
                if (arms[a].fn)
                    arms[a].fn(&dvs[j], (uint64_t)d_pts, n, 0, 0, 0, 0);
                else if (a == 1)
                    osn_tsdf_cuda_allocate_no_cas(vols[j], d_pts, n, 0, 0, 0, 0);
                else
                    osn_tsdf_cuda_allocate_blocks(vols[j], d_pts, n, 0, 0, 0, 0);
            }
            const double ms = t.stop_ms() / BATCH;
            if (i >= 5) samples[a].push_back(ms);
        }
    }

    const double a3 = median(samples[0]);
    const double a4 = median(samples[2]);
    std::printf("  %-11s %-32s %9s %8s %s\n", "arm", "what", "p50 ms", "x/A3", "recovered");
    std::printf("  ------------------------------------------------------------------------------\n");
    for (int a = 0; a < 9; ++a) {
        const double m = median(samples[a]);
        // How much of the A4-minus-A3 excess this variant gives back.
        const double recovered = (a4 - a3) > 0 ? 100.0 * (a4 - m) / (a4 - a3) : 0.0;
        char rec[32] = "";
        if (a >= 3) std::snprintf(rec, sizeof(rec), "%.0f%%", recovered);
        std::printf("  %-11s %-32s %9.4f %8.2f %s\n", arms[a].name, arms[a].what, m, m / a3, rec);
    }

    std::printf("\n  \"recovered\" is the share of A4's excess over A3 that disabling this\n");
    std::printf("  component gives back. A variant near 100%% contains the cost; one near\n");
    std::printf("  0%% does not. These variants compute the WRONG answer by construction.\n");

    for (auto* v : vols) osn_tsdf_cuda_destroy(v);
    cudaFree(d_pts);
    return 0;
}
