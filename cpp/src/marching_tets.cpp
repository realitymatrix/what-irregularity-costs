// Surface extraction by marching tetrahedra.
//
// Deliberately not marching cubes. MC needs a 256-entry case table plus a
// 256x16 triangle table; both are transcribed rather than derived, and a single
// wrong entry produces a hole in one specific corner configuration that no
// aggregate metric reliably catches. MT derives its cases from first
// principles: a tetrahedron has 4 corners, so 16 sign patterns, each of which
// is empty, one triangle, or a quad split into two. That fits on a screen and
// can be reasoned about.
//
// It is also watertight by construction. Adjacent tetrahedra share a face, and
// a vertex on a shared edge is interpolated from the same two corner values on
// both sides, so it lands at bitwise the same position. No cracks.
//
// Cost: about 2x the triangles of MC for the same surface. Accepted. Every arm
// uses this same extractor, so the comparison is unaffected, and correctness
// that can be verified by reading beats a table that cannot.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

#include "osn_tsdf/volume.hpp"

namespace osn_tsdf {
namespace {

/// The 6 tetrahedra tiling a cube, as indices into the 8 corners.
///
/// Every tetrahedron includes corners 0 and 7 (the main diagonal), which is
/// what makes the decomposition consistent between neighbouring cubes: the
/// shared diagonal means two cubes meeting at a face split that face the same
/// way, so their triangles line up.
constexpr int kTets[6][4] = {
    {0, 5, 1, 7}, {0, 1, 3, 7}, {0, 3, 2, 7},
    {0, 6, 4, 7}, {0, 4, 5, 7}, {0, 2, 6, 7},
};

/// Cube corner offsets, in the standard bit order (x = bit0, y = bit1, z = bit2).
constexpr int kCorner[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1},
};

struct Vertex {
    float p[3];
    float c[3];
};

/// Linear interpolation to the iso crossing between two corners.
///
/// Ordered by position so that the same edge, visited from either adjacent
/// tetrahedron, yields bitwise identical coordinates. Without that ordering the
/// two sides compute `a + t*(b-a)` and `b + t'*(a-b)`, which differ in the last
/// ulp and open sub-micron cracks that later show up as a non-watertight mesh.
Vertex interp(float iso, const float pa[3], const float pb[3], float va, float vb,
              const float ca[3], const float cb[3]) {
    const bool swap = (pb[0] < pa[0]) || (pb[0] == pa[0] && pb[1] < pa[1]) ||
                      (pb[0] == pa[0] && pb[1] == pa[1] && pb[2] < pa[2]);
    const float* p0 = swap ? pb : pa;
    const float* p1 = swap ? pa : pb;
    const float* c0 = swap ? cb : ca;
    const float* c1 = swap ? ca : cb;
    const float v0 = swap ? vb : va;
    const float v1 = swap ? va : vb;

    const float denom = v1 - v0;
    const float t = std::fabs(denom) < 1e-12f ? 0.5f : (iso - v0) / denom;
    const float tc = std::clamp(t, 0.0f, 1.0f);

    Vertex out{};
    for (int k = 0; k < 3; ++k) {
        out.p[k] = p0[k] + tc * (p1[k] - p0[k]);
        out.c[k] = c0[k] + tc * (c1[k] - c0[k]);
    }
    return out;
}

}  // namespace

int32_t Volume::extract_mesh(const MeshBuffers& out, float min_weight, float iso) const {
    if (out.capacity_vertices <= 0) return -1;

    const int32_t n_blocks = block_count_.load(std::memory_order_acquire);
    const float vs = cfg_.voxel_size_m;

    std::atomic<int32_t> n_written{0};
    std::atomic<bool> overflow{false};

    const int n_threads = n_threads_;
    const int32_t chunk = (n_blocks + n_threads - 1) / std::max(1, n_threads);

    auto worker = [&](int32_t lo, int32_t hi) {
        std::array<Vertex, 6> tri{};
        for (int32_t bi = lo; bi < hi; ++bi) {
            const BlockCoord bc = block_coord_[static_cast<std::size_t>(bi)];

            // One extra voxel on the far side so cubes spanning the block
            // boundary are emitted exactly once, by the lower block. Omitting
            // it leaves a one-voxel gap on every block face, which looks like
            // a shattered mesh rather than an off-by-one.
            for (int32_t lz = 0; lz < kBlockDim; ++lz) {
                for (int32_t ly = 0; ly < kBlockDim; ++ly) {
                    for (int32_t lx = 0; lx < kBlockDim; ++lx) {
                        const int32_t vx = bc.x * kBlockDim + lx;
                        const int32_t vy = bc.y * kBlockDim + ly;
                        const int32_t vz = bc.z * kBlockDim + lz;

                        float val[8];
                        float pos[8][3];
                        float col[8][3];
                        bool ok = true;
                        for (int c = 0; c < 8 && ok; ++c) {
                            float rgb[3];
                            ok = sample(vx + kCorner[c][0], vy + kCorner[c][1],
                                        vz + kCorner[c][2], min_weight, val[c], rgb);
                            if (!ok) break;
                            // Voxel CENTRES. Points are binned with
                            // floor(p / voxel), so voxel v spans
                            // [v*vs, (v+1)*vs) and its sample sits at
                            // (v + 0.5)*vs. Extracting at v*vs would shift the
                            // whole surface by half a voxel.
                            pos[c][0] = (static_cast<float>(vx + kCorner[c][0]) + 0.5f) * vs;
                            pos[c][1] = (static_cast<float>(vy + kCorner[c][1]) + 0.5f) * vs;
                            pos[c][2] = (static_cast<float>(vz + kCorner[c][2]) + 0.5f) * vs;
                            col[c][0] = rgb[0]; col[c][1] = rgb[1]; col[c][2] = rgb[2];
                        }
                        // Any missing corner means the cube straddles
                        // unobserved space. Skipping is what stops the surface
                        // from closing over holes it never saw.
                        if (!ok) continue;

                        for (const auto& tet : kTets) {
                            const int i0 = tet[0], i1 = tet[1], i2 = tet[2], i3 = tet[3];
                            const int code = (val[i0] < iso ? 1 : 0) | (val[i1] < iso ? 2 : 0) |
                                             (val[i2] < iso ? 4 : 0) | (val[i3] < iso ? 8 : 0);
                            int n_tri = 0;

                            auto emit = [&](int a, int b, int c, int d) {
                                // Crossings on the three edges from `a` into
                                // the other-signed corners.
                                tri[n_tri * 3 + 0] =
                                    interp(iso, pos[a], pos[b], val[a], val[b], col[a], col[b]);
                                tri[n_tri * 3 + 1] =
                                    interp(iso, pos[a], pos[c], val[a], val[c], col[a], col[c]);
                                tri[n_tri * 3 + 2] =
                                    interp(iso, pos[a], pos[d], val[a], val[d], col[a], col[d]);
                                ++n_tri;
                            };
                            auto emit_quad = [&](int a, int b, int c, int d) {
                                // a,b same sign; c,d the other. The crossing is
                                // a quad on edges ac, ad, bd, bc; split it.
                                const Vertex ac = interp(iso, pos[a], pos[c], val[a], val[c], col[a], col[c]);
                                const Vertex ad = interp(iso, pos[a], pos[d], val[a], val[d], col[a], col[d]);
                                const Vertex bd = interp(iso, pos[b], pos[d], val[b], val[d], col[b], col[d]);
                                const Vertex bc = interp(iso, pos[b], pos[c], val[b], val[c], col[b], col[c]);
                                tri[0] = ac; tri[1] = ad; tri[2] = bd;
                                tri[3] = ac; tri[4] = bd; tri[5] = bc;
                                n_tri = 2;
                            };

                            switch (code) {
                                case 0x00: case 0x0F: break;  // no crossing
                                case 0x01: emit(i0, i1, i2, i3); break;
                                case 0x0E: emit(i0, i1, i2, i3); break;
                                case 0x02: emit(i1, i0, i2, i3); break;
                                case 0x0D: emit(i1, i0, i2, i3); break;
                                case 0x04: emit(i2, i0, i1, i3); break;
                                case 0x0B: emit(i2, i0, i1, i3); break;
                                case 0x08: emit(i3, i0, i1, i2); break;
                                case 0x07: emit(i3, i0, i1, i2); break;
                                case 0x03: emit_quad(i0, i1, i2, i3); break;
                                case 0x0C: emit_quad(i2, i3, i0, i1); break;
                                case 0x05: emit_quad(i0, i2, i1, i3); break;
                                case 0x0A: emit_quad(i1, i3, i0, i2); break;
                                case 0x06: emit_quad(i1, i2, i0, i3); break;
                                case 0x09: emit_quad(i0, i3, i1, i2); break;
                                default: break;
                            }
                            if (n_tri == 0) continue;

                            const int32_t n_v = n_tri * 3;
                            const int32_t base =
                                n_written.fetch_add(n_v, std::memory_order_acq_rel);
                            if (base + n_v > out.capacity_vertices) {
                                overflow.store(true, std::memory_order_release);
                                continue;
                            }
                            for (int32_t v = 0; v < n_v; ++v) {
                                const Vertex& s = tri[static_cast<std::size_t>(v)];
                                float* dst = out.posnor + static_cast<std::size_t>(base + v) * 6;
                                dst[0] = s.p[0]; dst[1] = s.p[1]; dst[2] = s.p[2];
                                // Face normal, computed once per triangle.
                                dst[3] = 0.0f; dst[4] = 0.0f; dst[5] = 0.0f;
                                if (out.rgb) {
                                    uint8_t* c = out.rgb + static_cast<std::size_t>(base + v) * 3;
                                    c[0] = static_cast<uint8_t>(std::clamp(s.c[0], 0.0f, 255.0f));
                                    c[1] = static_cast<uint8_t>(std::clamp(s.c[1], 0.0f, 255.0f));
                                    c[2] = static_cast<uint8_t>(std::clamp(s.c[2], 0.0f, 255.0f));
                                }
                            }
                            // Face normal from the emitted triangles.
                            for (int32_t t = 0; t < n_tri; ++t) {
                                float* v0 = out.posnor + static_cast<std::size_t>(base + t * 3) * 6;
                                float* v1 = v0 + 6;
                                float* v2 = v0 + 12;
                                const float e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
                                const float e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
                                float nx = e1[1] * e2[2] - e1[2] * e2[1];
                                float ny = e1[2] * e2[0] - e1[0] * e2[2];
                                float nz = e1[0] * e2[1] - e1[1] * e2[0];
                                const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
                                if (len > 1e-20f) { nx /= len; ny /= len; nz /= len; }
                                for (float* v : {v0, v1, v2}) { v[3] = nx; v[4] = ny; v[5] = nz; }
                            }
                        }
                    }
                }
            }
        }
    };

    if (n_threads <= 1 || n_blocks <= 0) {
        worker(0, n_blocks);
    } else {
        std::vector<std::thread> pool;
        pool.reserve(static_cast<std::size_t>(n_threads) - 1);
        for (int t = 1; t < n_threads; ++t) {
            const int32_t lo = std::min(n_blocks, chunk * t);
            const int32_t hi = std::min(n_blocks, lo + chunk);
            if (lo >= hi) break;
            pool.emplace_back([&worker, lo, hi] { worker(lo, hi); });
        }
        worker(0, std::min(n_blocks, chunk));
        for (auto& th : pool) th.join();
    }

    if (overflow.load(std::memory_order_acquire)) return -1;
    return n_written.load(std::memory_order_acquire);
}

}  // namespace osn_tsdf
