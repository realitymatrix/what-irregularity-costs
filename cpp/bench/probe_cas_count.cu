// Dynamic compare-exchange count: A3 against A4 on identical input.
//
// Static SASS puts the two allocate kernels at 1.02x instruction parity (520
// against 528) while A4 runs 5.9x slower, so the difference must be in how
// often instructions execute. Both arms tally every CAS ATTEMPT into
// `drop_count` under an identical compile-time switch.
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "analytic_scenes.hpp"
#include "osn_tsdf/c_api.h"

extern "C" {
int32_t a4_init();
int32_t a4_allocate_countcas(const OsnTsdfDeviceView*, uint64_t, int32_t, float, float, float, float);
}

int main(int argc, char** argv) {
    const int nt = argc > 1 ? atoi(argv[1]) : 100, np = argc > 2 ? atoi(argv[2]) : 200;
    cudaSetDevice(0); cudaFree(nullptr);
    if (a4_init() != 0) { printf("A4 unavailable\n"); return 1; }
    const auto pts = scenes::sphere(0.5f, nt, np);
    const int32_t n = (int32_t)(pts.size() / 3);
    float* d = nullptr;
    cudaMalloc(&d, pts.size() * sizeof(float));
    cudaMemcpy(d, pts.data(), pts.size() * sizeof(float), cudaMemcpyHostToDevice);

    printf("=== dynamic CAS attempts, %d points ===\n\n", n);
    printf("%-10s %14s %10s %14s\n", "arm", "CAS attempts", "blocks", "attempts/block");
    printf("--------------------------------------------------------\n");
    unsigned long long base = 0;
    for (int arm = 0; arm < 2; ++arm) {
        OsnTsdfCudaVolume* v = osn_tsdf_cuda_create(0.01f, 1 << 14, -1.0f, 32.0f);
        OsnTsdfDeviceView dv{};
        osn_tsdf_cuda_device_view(v, &dv);
        if (arm == 0) osn_tsdf_cuda_allocate_counting_cas(v, d, n, 0, 0, 0, 0);
        else a4_allocate_countcas(&dv, (uint64_t)d, n, 0, 0, 0, 0);
        osn_tsdf_cuda_synchronize(v);
        const unsigned long long c = osn_tsdf_cuda_drop_count(v);
        const int32_t b = osn_tsdf_cuda_block_count(v);
        printf("%-10s %14llu %10d %14.1f\n", arm ? "A4 rust" : "A3 cuda", c, b,
               b ? (double)c / b : 0.0);
        if (arm == 0) base = c;
        else if (base) printf("\n  A4 issues %.2fx as many compare-exchange attempts as A3.\n",
                              (double)c / base);
        osn_tsdf_cuda_destroy(v);
    }
    cudaFree(d);
    return 0;
}
