#include "osn_tsdf/c_api.h"

#include <new>

#include "osn_tsdf/volume.hpp"

using osn_tsdf::MeshBuffers;
using osn_tsdf::PointBatch;
using osn_tsdf::Volume;
using osn_tsdf::VolumeConfig;

struct OsnTsdfVolume {
    Volume vol;
    explicit OsnTsdfVolume(const VolumeConfig& c) : vol(c) {}
};

namespace {
PointBatch make_batch(const float* pos, const uint8_t* col, const float* w, int32_t n,
                      int32_t chunk, float cx, float cy, float cz, float radius) {
    PointBatch b;
    b.positions = pos;
    b.colors = col;
    b.weights = w;
    b.n = n;
    b.chunk_id = chunk;
    b.cam[0] = cx; b.cam[1] = cy; b.cam[2] = cz;
    b.radius_m = radius;
    return b;
}
}  // namespace

extern "C" {

OsnTsdfVolume* osn_tsdf_create(float voxel_size_m, int32_t pool_capacity_blocks, float trunc_m,
                               float weight_cap) {
    VolumeConfig cfg;
    cfg.voxel_size_m = voxel_size_m;
    cfg.pool_capacity_blocks = pool_capacity_blocks;
    cfg.trunc_m = trunc_m;
    cfg.weight_cap = weight_cap;
    // Exceptions must not cross the C boundary; report failure as NULL, which
    // is what the header documents.
    try {
        return new OsnTsdfVolume(cfg);
    } catch (...) {
        return nullptr;
    }
}

void osn_tsdf_destroy(OsnTsdfVolume* v) { delete v; }

int32_t osn_tsdf_integrate(OsnTsdfVolume* v, const float* positions, const uint8_t* colors,
                           const float* weights, int32_t n_points, int32_t chunk_id, float cam_x,
                           float cam_y, float cam_z, float radius_m) {
    if (!v || !positions || n_points <= 0) return 1;
    try {
        v->vol.integrate(make_batch(positions, colors, weights, n_points, chunk_id, cam_x, cam_y,
                                    cam_z, radius_m));
        return 0;
    } catch (...) {
        return 2;
    }
}

int32_t osn_tsdf_allocate_blocks(OsnTsdfVolume* v, const float* positions, int32_t n_points,
                                 float cam_x, float cam_y, float cam_z, float radius_m) {
    if (!v || !positions || n_points <= 0) return 1;
    try {
        v->vol.allocate_blocks(
            make_batch(positions, nullptr, nullptr, n_points, 0, cam_x, cam_y, cam_z, radius_m));
        return 0;
    } catch (...) {
        return 2;
    }
}

int32_t osn_tsdf_update_voxels(OsnTsdfVolume* v, const float* positions, const uint8_t* colors,
                               const float* weights, int32_t n_points, int32_t chunk_id,
                               float cam_x, float cam_y, float cam_z, float radius_m) {
    if (!v || !positions || n_points <= 0) return 1;
    try {
        v->vol.update_voxels(make_batch(positions, colors, weights, n_points, chunk_id, cam_x,
                                        cam_y, cam_z, radius_m));
        return 0;
    } catch (...) {
        return 2;
    }
}

int32_t osn_tsdf_extract_mesh(OsnTsdfVolume* v, float* buffer_posnor, uint8_t* buffer_rgb,
                              int32_t vert_cap, float min_weight, float iso) {
    if (!v || !buffer_posnor || vert_cap <= 0) return -1;
    MeshBuffers out;
    out.posnor = buffer_posnor;
    out.rgb = buffer_rgb;
    out.capacity_vertices = vert_cap;
    try {
        return v->vol.extract_mesh(out, min_weight, iso);
    } catch (...) {
        return -1;
    }
}

int32_t osn_tsdf_block_count(const OsnTsdfVolume* v) { return v ? v->vol.block_count() : 0; }
uint64_t osn_tsdf_drop_count(const OsnTsdfVolume* v) { return v ? v->vol.drop_count() : 0; }
void osn_tsdf_set_thread_count(OsnTsdfVolume* v, int32_t n) {
    if (v) v->vol.set_thread_count(static_cast<int>(n));
}
int32_t osn_tsdf_thread_count(const OsnTsdfVolume* v) {
    return v ? static_cast<int32_t>(v->vol.thread_count()) : 0;
}

}  // extern "C"
