#include "osn_tsdf/c_api.h"

#include "cuda_impl.cuh"

using osn_tsdf::CudaVolume;
using osn_tsdf::CudaVolumeImpl;
using osn_tsdf::MeshBuffers;
using osn_tsdf::PointBatch;
using osn_tsdf::VolumeConfig;

// Guard the ABI mirrored by crates/tsdf-rust-cuda. A size change means a field
// was added or reordered, which silently shifts every field after it for any
// consumer that was not updated in the same commit.
static_assert(sizeof(OsnTsdfDeviceView) == 104,
              "OsnTsdfDeviceView layout changed: update crates/tsdf-rust-cuda DeviceViewC "
              "and the Triton argument list in cpp/bench and cpp/tests");

struct OsnTsdfCudaVolume {
    CudaVolume vol;
    explicit OsnTsdfCudaVolume(const VolumeConfig& c) : vol(c) {}
};

namespace {
PointBatch batch(const float* pos, int32_t n, float cx, float cy, float cz, float radius) {
    PointBatch b;
    b.positions = pos;
    b.n = n;
    b.cam[0] = cx; b.cam[1] = cy; b.cam[2] = cz;
    b.radius_m = radius;
    return b;
}
}  // namespace

extern "C" {

OsnTsdfCudaVolume* osn_tsdf_cuda_create(float voxel_size_m, int32_t pool_capacity_blocks,
                                        float trunc_m, float weight_cap) {
    VolumeConfig cfg;
    cfg.voxel_size_m = voxel_size_m;
    cfg.pool_capacity_blocks = pool_capacity_blocks;
    cfg.trunc_m = trunc_m;
    cfg.weight_cap = weight_cap;
    auto* v = new OsnTsdfCudaVolume(cfg);
    if (!v->vol.valid()) { delete v; return nullptr; }
    return v;
}

void osn_tsdf_cuda_destroy(OsnTsdfCudaVolume* v) { delete v; }
int32_t osn_tsdf_cuda_valid(const OsnTsdfCudaVolume* v) { return v && v->vol.valid() ? 1 : 0; }

int32_t osn_tsdf_cuda_device_view(const OsnTsdfCudaVolume* v, OsnTsdfDeviceView* out) {
    if (!v || !out || !v->vol.valid()) return 1;
    const auto& dv = v->vol.device_view();
    out->table = dv.table;
    out->block_count = dv.block_count;
    out->drop_count = dv.drop_count;
    out->block_coord = dv.block_coord;
    out->tsdf = dv.tsdf;
    out->weight = dv.weight;
    out->r = dv.r;
    out->g = dv.g;
    out->b = dv.b;
    out->hash_mask = dv.hash_mask;
    out->scratch_base = dv.scratch_base;
    out->pool_capacity = dv.pool_capacity;
    out->block_dim = osn_tsdf::kBlockDim;
    out->voxel_size_m = dv.voxel_size;
    out->trunc_m = dv.trunc;
    out->weight_cap = dv.weight_cap;
    return 0;
}

int32_t osn_tsdf_cuda_allocate_blocks(OsnTsdfCudaVolume* v, const float* d_positions,
                                      int32_t n_points, float cam_x, float cam_y, float cam_z,
                                      float radius_m) {
    if (!v || !d_positions || n_points <= 0) return 1;
    v->vol.allocate_blocks(batch(d_positions, n_points, cam_x, cam_y, cam_z, radius_m));
    return 0;
}

int32_t osn_tsdf_cuda_update_voxels(OsnTsdfCudaVolume* v, const float* d_positions,
                                    int32_t n_points, float cam_x, float cam_y, float cam_z,
                                    float radius_m) {
    if (!v || !d_positions || n_points <= 0) return 1;
    v->vol.update_voxels(batch(d_positions, n_points, cam_x, cam_y, cam_z, radius_m));
    return 0;
}

int32_t osn_tsdf_cuda_extract_mesh(OsnTsdfCudaVolume* v, float* buffer_posnor, uint8_t* buffer_rgb,
                                   int32_t vert_cap, float min_weight, float iso) {
    if (!v || !buffer_posnor || vert_cap <= 0) return -1;
    MeshBuffers out;
    out.posnor = buffer_posnor;
    out.rgb = buffer_rgb;
    out.capacity_vertices = vert_cap;
    return v->vol.extract_mesh(out, min_weight, iso);
}

int32_t osn_tsdf_cuda_reset(OsnTsdfCudaVolume* v) {
    if (!v) return 1;
    v->vol.reset();
    return 0;
}

int32_t osn_tsdf_cuda_block_count(const OsnTsdfCudaVolume* v) { return v ? v->vol.block_count() : 0; }
uint64_t osn_tsdf_cuda_drop_count(const OsnTsdfCudaVolume* v) { return v ? v->vol.drop_count() : 0; }
void osn_tsdf_cuda_synchronize(const OsnTsdfCudaVolume* v) { if (v) v->vol.synchronize(); }

}  // extern "C"
