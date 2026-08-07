// Does `atom.cas.b128` work at all through cuda-oxide's `ptx_asm!`?
//
// One thread, one 16-byte cell preset to (-1, -1, 0), one exchange to (42, 7).
// The 128-bit insert saturated the pool and dropped every point, which is what
// a compare-exchange that never matches looks like from outside, so this checks
// the instruction in isolation before blaming the algorithm.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
extern "C" {
int32_t a4_init();
int32_t a4_cas128_selftest(uint64_t cell, uint64_t out);
}
int main() {
    cudaSetDevice(0); cudaFree(nullptr);
    if (a4_init() != 0) { printf("A4 unavailable\n"); return 1; }
    int64_t host_cell[2];
    // key = -1, block_idx = -1, pad = 0
    host_cell[0] = -1;
    ((int32_t*)&host_cell[1])[0] = -1;
    ((int32_t*)&host_cell[1])[1] = 0;
    int64_t *d_cell = nullptr, *d_out = nullptr;
    cudaMalloc(&d_cell, 16); cudaMalloc(&d_out, 32);
    cudaMemcpy(d_cell, host_cell, 16, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, 32);
    const int rc = a4_cas128_selftest((uint64_t)d_cell, (uint64_t)d_out);
    cudaDeviceSynchronize();
    int64_t out[4] = {0,0,0,0};
    cudaMemcpy(out, d_out, 32, cudaMemcpyDeviceToHost);
    printf("launch rc            %d\n", rc);
    printf("expected  lo/hi      %#llx / %#llx\n", (unsigned long long)-1LL, 0xFFFFFFFFull);
    printf("returned  lo/hi      %#llx / %#llx\n", (unsigned long long)out[0], (unsigned long long)out[1]);
    printf("cell after lo/hi     %#llx / %#llx\n", (unsigned long long)out[2], (unsigned long long)out[3]);
    printf("desired   lo/hi      %#llx / %#llx\n", 42ull, 7ull);
    const bool matched = (unsigned long long)out[0] == 0xFFFFFFFFFFFFFFFFull;
    const bool wrote = out[2] == 42 && (out[3] & 0xFFFFFFFF) == 7;
    printf("\n  exchange matched:  %s\n  memory updated:    %s\n",
           matched ? "YES" : "NO", wrote ? "YES" : "NO");
    return 0;
}
