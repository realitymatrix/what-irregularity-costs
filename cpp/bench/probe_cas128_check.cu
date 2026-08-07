#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "analytic_scenes.hpp"
#include "osn_tsdf/c_api.h"
extern "C" {
int32_t a4_init();
int32_t a4_allocate_blocks(const OsnTsdfDeviceView*, uint64_t, int32_t, float, float, float, float);
int32_t a4_allocate_cas128(const OsnTsdfDeviceView*, uint64_t, int32_t, float, float, float, float);
int32_t a4_allocate_slotidx(const OsnTsdfDeviceView*, uint64_t, int32_t, float, float, float, float);
}
int main(int argc, char** argv) {
    const int nt = argc > 1 ? atoi(argv[1]) : 400, np = argc > 2 ? atoi(argv[2]) : 800;
    cudaSetDevice(0); cudaFree(nullptr);
    if (a4_init() != 0) { printf("A4 unavailable\n"); return 1; }
    const auto pts = scenes::sphere(0.5f, nt, np);
    const int32_t n = (int32_t)(pts.size() / 3);
    float* d = nullptr;
    cudaMalloc(&d, pts.size()*sizeof(float));
    cudaMemcpy(d, pts.data(), pts.size()*sizeof(float), cudaMemcpyHostToDevice);
    printf("=== allocate correctness, %d points ===\n%-14s %10s %10s\n", n, "arm", "blocks", "drops");
    printf("--------------------------------------------------------\n");
    std::vector<float> mesh((size_t)(4<<20)*6);
    for (int arm = 0; arm < 4; ++arm) {
        OsnTsdfCudaVolume* v = osn_tsdf_cuda_create(0.01f, 1<<16, -1.0f, 32.0f);
        OsnTsdfDeviceView dv{}; osn_tsdf_cuda_device_view(v, &dv);
        if (arm == 0) osn_tsdf_cuda_allocate_blocks(v, d, n, 0,0,0,0);
        else if (arm == 1) a4_allocate_blocks(&dv, (uint64_t)d, n, 0,0,0,0);
        else if (arm == 2) a4_allocate_cas128(&dv, (uint64_t)d, n, 0,0,0,0);
        else a4_allocate_slotidx(&dv, (uint64_t)d, n, 0,0,0,0);
        osn_tsdf_cuda_synchronize(v);
        // Allocation only. The slot-index variant uses a different hash mask,
        // so the shared update kernel cannot locate its blocks; pairing them
        // would need a matching update, which is out of scope for measuring
        // allocation.
        const int32_t nv = -1;
        const char* nm = arm==0?"A3 cuda":arm==1?"A4 64-bit CAS":arm==2?"A4 128-bit CAS":"A4 slot-index";
        printf("%-14s %10d %10llu\n", nm,
               osn_tsdf_cuda_block_count(v), (unsigned long long)osn_tsdf_cuda_drop_count(v));
        (void)nv;
        osn_tsdf_cuda_destroy(v);
    }
    cudaFree(d); return 0;
}
