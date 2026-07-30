// Hash-blocked TSDF volume, CUDA implementation (arm A3).
//
// Deliberately the same algorithm as the CPU arm, not merely the same output:
// same hash function, same open-addressed table, same compare-exchange
// insertion protocol, same weighted-sum accumulation, same marching-tetrahedra
// extractor. The comparison is only meaningful if the arms differ in where the
// work runs and how it is expressed, not in what work is done.
//
// A3 is also the performance reference point the Rust (A4) and Triton (A5)
// arms are measured against, so anything here that is a deliberate
// optimisation rather than a faithful port has to be visible.
#pragma once

#include <cstdint>

#include "osn_tsdf/types.hpp"

namespace osn_tsdf {

/// Opaque device state. Defined in the .cu so this header stays includable
/// from plain C++ translation units that nvcc never sees.
struct CudaVolumeImpl;

class CudaVolume {
public:
    explicit CudaVolume(const VolumeConfig& cfg);
    ~CudaVolume();

    CudaVolume(const CudaVolume&) = delete;
    CudaVolume& operator=(const CudaVolume&) = delete;
    CudaVolume(CudaVolume&&) = delete;
    CudaVolume& operator=(CudaVolume&&) = delete;

    /// True when construction obtained all device memory. Construction does not
    /// throw across the C boundary, so callers must check.
    bool valid() const;

    /// Points are DEVICE pointers, world-space.
    ///
    /// Device-resident on purpose: the depth stage produces points on device
    /// (libinfer `infer_device_io`), so a host round-trip here would be an
    /// artefact of the harness rather than of the pipeline, and it is exactly
    /// the cost the host-transfer ablation is meant to isolate.
    void integrate(const PointBatch& batch);
    void allocate_blocks(const PointBatch& batch);
    void update_voxels(const PointBatch& batch);

    /// `out.posnor` / `out.rgb` are HOST buffers; the mesh is copied back.
    /// Returns the vertex count, or -1 on overflow.
    int32_t extract_mesh(const MeshBuffers& out, float min_weight, float iso) const;

    int32_t block_count() const;
    uint64_t drop_count() const;
    const VolumeConfig& config() const;

    /// Block the calling thread until queued work completes. Required before
    /// timing anything: launches are asynchronous, so a measurement that does
    /// not synchronise records queue-submission time, not execution time.
    void synchronize() const;

private:
    CudaVolumeImpl* impl_ = nullptr;
};

}  // namespace osn_tsdf
