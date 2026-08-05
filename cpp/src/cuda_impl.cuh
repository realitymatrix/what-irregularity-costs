// Private CUDA state, shared between the volume and the extractor.
// Not installed: nothing outside this library needs it, and exposing device
// pointers would invite callers to bypass the stream discipline.
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#include "osn_tsdf/volume_cuda.hpp"

namespace osn_tsdf {

#define CUDA_TRY(expr)                                                                  \
    do {                                                                                \
        const cudaError_t err_ = (expr);                                                \
        if (err_ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "[osn_tsdf] %s:%d: %s -> %s\n", __FILE__, __LINE__,    \
                         #expr, cudaGetErrorString(err_));                              \
            return false;                                                               \
        }                                                                               \
    } while (0)

/// Scratch slots appended to the hash table, one per lane of a Triton program.
inline constexpr uint32_t kScratchSlots = 256;

struct DeviceView {  // NOLINT: shared with other arms via the C API
    uint32_t scratch_base = 0;
    HashEntry* table;
    uint32_t hash_mask;
    int32_t* block_count;
    unsigned long long* drop_count;
    BlockCoord* block_coord;
    float* tsdf;
    float* weight;
    float* r;
    float* g;
    float* b;
    int32_t pool_capacity;
    float voxel_size;
    float trunc;
    float weight_cap;
};


struct CudaVolumeImpl {
    VolumeConfig cfg;
    DeviceView v{};
    bool ok = false;
    cudaStream_t stream = nullptr;

    bool alloc() {
        uint32_t size = 2u;
        while (size < (uint32_t)cfg.pool_capacity_blocks * 2u) size <<= 1;
        v.hash_mask = size - 1;
        v.pool_capacity = cfg.pool_capacity_blocks;
        v.voxel_size = cfg.voxel_size_m;
        v.trunc = cfg.trunc();
        v.weight_cap = cfg.weight_cap;

        const size_t n_vox = (size_t)cfg.pool_capacity_blocks * kBlockVoxels;
        CUDA_TRY(cudaStreamCreate(&stream));
        // Extra slots past the live table form a scratch REGION for the
        // Triton arm: tl.atomic_cas takes no mask, so resolved lanes must aim
        // their inert CAS somewhere, and one address per lane avoids
        // serialising the grid. Harmless for the other arms, which never
        // address past hash_mask.
        v.scratch_base = size;
        CUDA_TRY(cudaMalloc(&v.table, (size_t)(size + kScratchSlots) * sizeof(HashEntry)));
        CUDA_TRY(cudaMalloc(&v.block_count, sizeof(int32_t)));
        CUDA_TRY(cudaMalloc(&v.drop_count, sizeof(unsigned long long)));
        CUDA_TRY(cudaMalloc(&v.block_coord, (size_t)cfg.pool_capacity_blocks * sizeof(BlockCoord)));
        CUDA_TRY(cudaMalloc(&v.tsdf, n_vox * sizeof(float)));
        CUDA_TRY(cudaMalloc(&v.weight, n_vox * sizeof(float)));
        CUDA_TRY(cudaMalloc(&v.r, n_vox * sizeof(float)));
        CUDA_TRY(cudaMalloc(&v.g, n_vox * sizeof(float)));
        CUDA_TRY(cudaMalloc(&v.b, n_vox * sizeof(float)));

        // The table must be filled with kEmptyKey, not zeroed: zero is a valid
        // block coordinate, so a zeroed table reads as occupied at (0,0,0).
        std::vector<HashEntry> host(size + kScratchSlots, HashEntry{kEmptyKey, -1, 0});
        // Scratch sentinels must differ from kEmptyKey and from any packed key,
        // so an inert CAS can never succeed there.
        for (uint32_t i = size; i < size + kScratchSlots; ++i) host[i].key = -2;
        CUDA_TRY(cudaMemcpy(v.table, host.data(),
                            (size_t)(size + kScratchSlots) * sizeof(HashEntry),
                            cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemset(v.block_count, 0, sizeof(int32_t)));
        CUDA_TRY(cudaMemset(v.drop_count, 0, sizeof(unsigned long long)));
        CUDA_TRY(cudaMemset(v.tsdf, 0, n_vox * sizeof(float)));
        CUDA_TRY(cudaMemset(v.weight, 0, n_vox * sizeof(float)));
        CUDA_TRY(cudaMemset(v.r, 0, n_vox * sizeof(float)));
        CUDA_TRY(cudaMemset(v.g, 0, n_vox * sizeof(float)));
        CUDA_TRY(cudaMemset(v.b, 0, n_vox * sizeof(float)));
        ok = true;
        return true;
    }

    ~CudaVolumeImpl() {
        cudaFree(v.table);
        cudaFree(v.block_count);
        cudaFree(v.drop_count);
        cudaFree(v.block_coord);
        cudaFree(v.tsdf);
        cudaFree(v.weight);
        cudaFree(v.r);
        cudaFree(v.g);
        cudaFree(v.b);
        if (stream) cudaStreamDestroy(stream);
    }
};


}  // namespace osn_tsdf
