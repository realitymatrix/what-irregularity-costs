// Arm A3 (CUDA C++) against closed-form geometry, and against arm A2.
//
// Tiers 1 and 3 of the correctness design meeting for the first time:
//
//   Tier 1, analytic. A3 is measured against mathematics. This is the tier
//   that can catch a mistake both arms share, which is why it comes first.
//
//   Tier 3, cross-arm. A2 and A3 are fed bit-identical points and their
//   outputs compared directly. This catches per-arm bugs that the analytic
//   test is too coarse to see, but by construction it cannot catch anything
//   the two arms get wrong together.
//
// The cross-check is one-sided surface distance over a uniform grid, not a
// vertex-set comparison: the two arms accumulate in different orders, so
// individual vertices differ in the last ulp and the triangle soups are
// emitted in different orders. Geometry is what must agree, not bit patterns.

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include "analytic_scenes.hpp"
#include "osn_tsdf/c_api.h"
#include "osn_tsdf/volume.hpp"
#include "osn_tsdf/triton_launch.h"
#include "osn_tsdf/volume_cuda.hpp"

#if __has_include("triton_tsdf_manifest.h")
#include "triton_tsdf_manifest.h"
#define OSN_TSDF_HAVE_A5 1
#endif

#ifdef OSN_TSDF_HAVE_A4
// Arm A4 lives in a Rust staticlib built by cuda-oxide. It allocates nothing:
// it runs its kernels over the device state this library owns, then hands the
// volume back for the shared extraction. See docs/ARM-SCOPE.md.
extern "C" {
int32_t a4_init();
int32_t a4_allocate_blocks(const OsnTsdfDeviceView* view, uint64_t d_positions, int32_t n_points,
                           float cam_x, float cam_y, float cam_z, float radius_m);
int32_t a4_update_voxels(const OsnTsdfDeviceView* view, uint64_t d_positions, int32_t n_points,
                         float cam_x, float cam_y, float cam_z, float radius_m);
int32_t a4_synchronize();
}
#endif

using namespace osn_tsdf;

namespace {

int g_failures = 0;

void check(const char* label, bool pass, const std::string& detail) {
    if (!pass) ++g_failures;
    std::printf("  [%s] %s  %s\n", pass ? "PASS" : "FAIL", label, detail.c_str());
}

std::string fmt(const char* f, double a, double b = 0.0) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), f, a, b);
    return buf;
}

/// Vertex positions only, stripped from the (pos3, nor3) interleave.
std::vector<float> positions_of(const std::vector<float>& posnor) {
    const std::size_t nv = posnor.size() / 6;
    std::vector<float> out(nv * 3);
    for (std::size_t i = 0; i < nv; ++i) {
        out[i * 3 + 0] = posnor[i * 6 + 0];
        out[i * 3 + 1] = posnor[i * 6 + 1];
        out[i * 3 + 2] = posnor[i * 6 + 2];
    }
    return out;
}

/// Query stride for surface-distance cross-checks.
///
/// Comparing all ~850k vertices against all others, for every arm pair, is
/// quadratic in wall-clock and adds nothing: the samples are dense on a smooth
/// surface, so every 8th vertex characterises the agreement just as well. The
/// grid is still built from ALL points, so a query can still find its true
/// nearest neighbour; only the number of queries is reduced.
constexpr std::size_t kCrossCheckStride = 8;

/// Uniform-grid nearest-neighbour distances from `a` to `b`, in metres.
struct GridNN {
    std::unordered_map<long long, std::vector<int>> cells;
    const std::vector<float>* pts;
    float inv_cell;

    static long long key(int x, int y, int z) {
        return (long long)(x + 1'000'000) * 4'000'000'000LL +
               (long long)(y + 1'000'000) * 2'000'000LL + (z + 1'000'000);
    }

    GridNN(const std::vector<float>& p, float cell) : pts(&p), inv_cell(1.0f / cell) {
        for (std::size_t i = 0; i < p.size() / 3; ++i) {
            cells[key((int)std::floor(p[i * 3] * inv_cell), (int)std::floor(p[i * 3 + 1] * inv_cell),
                      (int)std::floor(p[i * 3 + 2] * inv_cell))]
                .push_back((int)i);
        }
    }

    float nearest(const float q[3]) const {
        const int kx = (int)std::floor(q[0] * inv_cell);
        const int ky = (int)std::floor(q[1] * inv_cell);
        const int kz = (int)std::floor(q[2] * inv_cell);
        const float cell = 1.0f / inv_cell;
        float best = 1e30f;
        // Expand shells until the best hit is provably inside the scanned
        // radius. Not conditioned on improving during a shell: a shell can be
        // empty while a later one is not, and keying the exit on improvement
        // fails to terminate.
        for (int r = 0; r <= 64; ++r) {
            for (int dx = -r; dx <= r; ++dx)
                for (int dy = -r; dy <= r; ++dy)
                    for (int dz = -r; dz <= r; ++dz) {
                        if (r > 0 && std::abs(dx) != r && std::abs(dy) != r && std::abs(dz) != r)
                            continue;
                        auto it = cells.find(key(kx + dx, ky + dy, kz + dz));
                        if (it == cells.end()) continue;
                        for (int i : it->second) {
                            const float ex = (*pts)[i * 3] - q[0];
                            const float ey = (*pts)[i * 3 + 1] - q[1];
                            const float ez = (*pts)[i * 3 + 2] - q[2];
                            best = std::min(best, ex * ex + ey * ey + ez * ez);
                        }
                    }
            if (best < 1e29f && std::sqrt(best) <= r * cell) break;
        }
        return std::sqrt(best);
    }
};

std::vector<float> extract_cpu(Volume& v, float min_w, int32_t cap = 4 << 20) {
    std::vector<float> buf((std::size_t)cap * 6);
    MeshBuffers out;
    out.posnor = buf.data();
    out.capacity_vertices = cap;
    const int32_t n = v.extract_mesh(out, min_w, 0.0f);
    if (n < 0) return {};
    buf.resize((std::size_t)n * 6);
    return buf;
}

std::vector<float> extract_gpu(CudaVolume& v, float min_w, int32_t cap = 4 << 20) {
    std::vector<float> buf((std::size_t)cap * 6);
    MeshBuffers out;
    out.posnor = buf.data();
    out.capacity_vertices = cap;
    const int32_t n = v.extract_mesh(out, min_w, 0.0f);
    if (n < 0) return {};
    buf.resize((std::size_t)n * 6);
    return buf;
}

/// Upload host points and hand back a device pointer the caller must free.
float* upload(const std::vector<float>& h) {
    float* d = nullptr;
    if (cudaMalloc(&d, h.size() * sizeof(float)) != cudaSuccess) return nullptr;
    cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}

}  // namespace

int main() {
    std::printf("=== A3 (CUDA C++) analytic correctness + A2 cross-check ===\n\n");

    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("  no CUDA device; skipping\n");
        return 0;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("  device: %s sm_%d%d\n\n", prop.name, prop.major, prop.minor);

    const float R = 0.5f;
    const float voxel = 0.01f;
    const auto pts = scenes::sphere(R, 400, 800);
    const int32_t n_pts = (int32_t)(pts.size() / 3);

    VolumeConfig cfg;
    cfg.voxel_size_m = voxel;
    cfg.pool_capacity_blocks = 1 << 14;

    // ---- A3 on the sphere -------------------------------------------------
    float* d_pts = upload(pts);
    if (!d_pts) { std::printf("  upload failed\n"); return 1; }

    CudaVolume gpu(cfg);
    if (!gpu.valid()) { std::printf("  CUDA volume allocation failed\n"); return 1; }
    PointBatch gb;
    gb.positions = d_pts;
    gb.n = n_pts;
    gpu.integrate(gb);
    gpu.synchronize();

    const auto gmesh = extract_gpu(gpu, 0.5f);
    const std::size_t g_nv = gmesh.size() / 6;
    std::printf("--- A3 sphere R=%.2f m, voxel %.3f m ---\n", R, voxel);
    std::printf("  blocks %d, dropped %llu, vertices %zu\n", gpu.block_count(),
                (unsigned long long)gpu.drop_count(), g_nv);

    check("A3 produced geometry", g_nv > 1000, fmt("%.0f vertices", (double)g_nv));
    check("A3 no points dropped", gpu.drop_count() == 0,
          fmt("%.0f dropped", (double)gpu.drop_count()));

    double g_mean = 0.0, g_max = 0.0;
    for (std::size_t i = 0; i < g_nv; ++i) {
        const double x = gmesh[i * 6], y = gmesh[i * 6 + 1], z = gmesh[i * 6 + 2];
        const double e = std::fabs(std::sqrt(x * x + y * y + z * z) - R);
        g_mean += e;
        g_max = std::max(g_max, e);
    }
    if (g_nv) g_mean /= g_nv;
    check("A3 mean |r - R| < voxel/4", g_mean < voxel * 0.25,
          fmt("%.6f m (voxel %.3f)", g_mean, voxel));
    check("A3 max |r - R| < voxel", g_max < voxel, fmt("%.6f m", g_max));

    // ---- A2 on the same points -------------------------------------------
    Volume cpu(cfg);
    PointBatch cb;
    cb.positions = pts.data();
    cb.n = n_pts;
    cpu.integrate(cb);
    const auto cmesh = extract_cpu(cpu, 0.5f);
    const std::size_t c_nv = cmesh.size() / 6;

    std::printf("\n--- A2 vs A3, identical input ---\n");
    std::printf("  A2: %d blocks, %zu vertices\n", cpu.block_count(), c_nv);
    std::printf("  A3: %d blocks, %zu vertices\n", gpu.block_count(), g_nv);

    // Block count is exact: allocation is driven by geometry alone, so any
    // difference means the arms disagree about which blocks the scene touches,
    // which is a bug regardless of what the meshes look like.
    check("block counts identical", cpu.block_count() == gpu.block_count(),
          fmt("%.0f vs %.0f", (double)cpu.block_count(), (double)gpu.block_count()));

    const double vert_rel =
        c_nv || g_nv ? std::fabs((double)c_nv - (double)g_nv) / std::max(c_nv, g_nv) : 0.0;
    check("vertex counts within 1%", vert_rel < 0.01, fmt("%.4f%% apart", vert_rel * 100.0));

    // Symmetric surface distance. Not a vertex-set comparison: the arms
    // accumulate in different orders, so vertices differ in the last ulp and
    // are emitted in different orders. Geometry is what has to agree.
    if (c_nv && g_nv) {
        const auto cpos = positions_of(cmesh);
        const auto gpos = positions_of(gmesh);
        GridNN c_grid(cpos, voxel * 2.0f);
        GridNN g_grid(gpos, voxel * 2.0f);

        double sum_ab = 0.0, max_ab = 0.0, sum_ba = 0.0, max_ba = 0.0;
        for (std::size_t i = 0; i < g_nv; i += kCrossCheckStride) {
            const float d = c_grid.nearest(&gpos[i * 3]);
            sum_ab += d;
            max_ab = std::max(max_ab, (double)d);
        }
        for (std::size_t i = 0; i < c_nv; i += kCrossCheckStride) {
            const float d = g_grid.nearest(&cpos[i * 3]);
            sum_ba += d;
            max_ba = std::max(max_ba, (double)d);
        }
        const double n_ab = (double)((g_nv + kCrossCheckStride - 1) / kCrossCheckStride);
        const double n_ba = (double)((c_nv + kCrossCheckStride - 1) / kCrossCheckStride);
        const double mean_sym = 0.5 * (sum_ab / n_ab + sum_ba / n_ba);
        const double hausdorff = std::max(max_ab, max_ba);
        std::printf("  mean surface distance %.9f m, hausdorff %.9f m\n", mean_sym, hausdorff);

        // Sub-voxel by a wide margin: these are two implementations of one
        // algorithm on identical input, so they should differ only by float
        // accumulation order, not by discretisation.
        check("A2/A3 mean surface distance < voxel/100", mean_sym < voxel * 0.01,
              fmt("%.9f m (voxel %.3f)", mean_sym, voxel));
        check("A2/A3 hausdorff < voxel/10", hausdorff < voxel * 0.1, fmt("%.9f m", hausdorff));
    }

#ifdef OSN_TSDF_HAVE_A5
    // ---- A5a / A5b: Triton kernels over the same device state -------------
    //
    // A5a runs A3's CUDA allocation then Triton's update. A5b runs Triton for
    // both passes. The difference between them is the control-flow tax priced
    // on the real workload rather than inferred from a synthetic test.
    for (int variant = 0; variant < 2; ++variant) {
        const bool full = (variant == 1);
        const char* label = full ? "A5b triton-full" : "A5a triton-shared";
        std::printf("\n--- %s ---\n", label);

        const std::string dir = std::string(getenv("OSN_TRITON_DIR")
                                                ? getenv("OSN_TRITON_DIR")
                                                : "../artifacts/triton");
        OsnTritonKernel* k_upd =
            osn_triton_load((dir + "/" + OSN_TRITON_UPDATE_NAME + ".cubin").c_str(),
                            OSN_TRITON_UPDATE_NAME);
        OsnTritonKernel* k_alloc =
            full ? osn_triton_load((dir + "/" + OSN_TRITON_ALLOC_NAME + ".cubin").c_str(),
                                   OSN_TRITON_ALLOC_NAME)
                 : nullptr;
        if (!k_upd || (full && !k_alloc)) {
            check(full ? "A5b cubin load" : "A5a cubin load", false, "missing artifacts");
        } else {
            OsnTsdfCudaVolume* tv =
                osn_tsdf_cuda_create(voxel, cfg.pool_capacity_blocks, -1.0f, 32.0f);
            OsnTsdfDeviceView dv{};
            osn_tsdf_cuda_device_view(tv, &dv);


            const int32_t n = n_pts;
            const int32_t hash_mask_i = (int32_t)dv.hash_mask;
            const float rsq = 0.0f, camv = 0.0f;
            const uint32_t grid = (uint32_t)((n + OSN_TRITON_BLOCK - 1) / OSN_TRITON_BLOCK);

            if (full) {
                // Historical scratch layout: BLOCK slots shared by the whole
                // grid. Pinned here rather than tracking the benchmark's
                // --scratch-slots, so a change to the region size is measured as
                // performance and cannot quietly change what is verified.
                const int32_t scratch_mask = OSN_TRITON_BLOCK - 1;
                void* aargs[] = {(void*)&d_pts,      (void*)&dv.table,     (void*)&dv.table,
                                 (void*)&dv.block_count, (void*)&dv.block_coord,
                                 (void*)&dv.drop_count,  (void*)&dv.scratch_base,
                                 (void*)&scratch_mask,
                                 (void*)&n,          (void*)&hash_mask_i,  (void*)&dv.pool_capacity,
                                 (void*)&dv.voxel_size_m, (void*)&dv.trunc_m,
                                 (void*)&camv, (void*)&camv, (void*)&camv, (void*)&rsq};
                osn_triton_launch(k_alloc, grid, OSN_TRITON_ALLOC_BLOCK_DIM_X,
                                  OSN_TRITON_ALLOC_SHARED, aargs,
                                  (int32_t)(sizeof(aargs) / sizeof(aargs[0])));
            } else {
                osn_tsdf_cuda_allocate_blocks(tv, d_pts, n, 0.0f, 0.0f, 0.0f, 0.0f);
                osn_tsdf_cuda_synchronize(tv);
            }

            void* uargs[] = {(void*)&d_pts,   (void*)&dv.table,   (void*)&dv.table,
                             (void*)&dv.tsdf, (void*)&dv.weight,
                             (void*)&dv.r, (void*)&dv.g, (void*)&dv.b, (void*)&n,
                             (void*)&hash_mask_i, (void*)&dv.voxel_size_m, (void*)&dv.trunc_m,
                             (void*)&dv.weight_cap, (void*)&camv, (void*)&camv, (void*)&camv,
                             (void*)&rsq};
            osn_triton_launch(k_upd, grid, OSN_TRITON_UPDATE_BLOCK_DIM_X,
                              OSN_TRITON_UPDATE_SHARED, uargs,
                              (int32_t)(sizeof(uargs) / sizeof(uargs[0])));
            cuCtxSynchronize();
            osn_tsdf_cuda_synchronize(tv);

            std::vector<float> tbuf((std::size_t)(4 << 20) * 6);
            const int32_t tn = osn_tsdf_cuda_extract_mesh(tv, tbuf.data(), nullptr, 4 << 20,
                                                          0.5f, 0.0f);
            const std::size_t t_nv = tn > 0 ? (std::size_t)tn : 0;
            tbuf.resize(t_nv * 6);
            std::printf("  blocks %d, vertices %zu\n", osn_tsdf_cuda_block_count(tv), t_nv);

            check((std::string(label) + " produced geometry").c_str(), t_nv > 1000,
                  fmt("%.0f vertices", (double)t_nv));
            check((std::string(label) + " block count matches A3").c_str(),
                  osn_tsdf_cuda_block_count(tv) == gpu.block_count(),
                  fmt("%.0f vs %.0f", (double)osn_tsdf_cuda_block_count(tv),
                      (double)gpu.block_count()));

            double tm = 0.0, tx = 0.0;
            for (std::size_t i = 0; i < t_nv; ++i) {
                const double x = tbuf[i * 6], y = tbuf[i * 6 + 1], z = tbuf[i * 6 + 2];
                const double e = std::fabs(std::sqrt(x * x + y * y + z * z) - R);
                tm += e; tx = std::max(tx, e);
            }
            if (t_nv) tm /= t_nv;
            check((std::string(label) + " mean |r - R| < voxel/4").c_str(), tm < voxel * 0.25,
                  fmt("%.6f m", tm));
            check((std::string(label) + " max |r - R| < voxel").c_str(), tx < voxel,
                  fmt("%.6f m", tx));

            if (t_nv && g_nv) {
                const auto tpos = positions_of(tbuf);
                const auto gpos3 = positions_of(gmesh);
                GridNN gg(gpos3, voxel * 2.0f);
                double s1 = 0.0, m1 = 0.0;
                for (std::size_t i = 0; i < t_nv; i += kCrossCheckStride) {
                    const float d = gg.nearest(&tpos[i * 3]);
                    s1 += d; m1 = std::max(m1, (double)d);
                }
                const double nq2 = (double)((t_nv + kCrossCheckStride - 1) / kCrossCheckStride);
                std::printf("  vs A3: mean %.9f m, hausdorff %.9f m\n", s1 / nq2, m1);
                check((std::string(label) + " vs A3 mean < voxel/100").c_str(),
                      s1 / nq2 < voxel * 0.01, fmt("%.9f m", s1 / nq2));
            }
            osn_tsdf_cuda_destroy(tv);
        }
        if (k_upd) osn_triton_unload(k_upd);
        if (k_alloc) osn_triton_unload(k_alloc);
    }
#endif

#ifdef OSN_TSDF_HAVE_A4
    // ---- A4: Rust kernels over C++-allocated device state -----------------
    {
        std::printf("\n--- A4 (Rust via cuda-oxide) on the same points ---\n");
        if (a4_init() != 0) {
            std::printf("  [FAIL] a4_init\n");
            ++g_failures;
        } else {
            OsnTsdfCudaVolume* rv = osn_tsdf_cuda_create(voxel, cfg.pool_capacity_blocks, -1.0f, 32.0f);
            OsnTsdfDeviceView dv{};
            if (!rv || osn_tsdf_cuda_device_view(rv, &dv) != 0) {
                std::printf("  [FAIL] device view\n");
                ++g_failures;
            } else {
                a4_allocate_blocks(&dv, (uint64_t)d_pts, n_pts, 0.0f, 0.0f, 0.0f, 0.0f);
                a4_update_voxels(&dv, (uint64_t)d_pts, n_pts, 0.0f, 0.0f, 0.0f, 0.0f);
                a4_synchronize();
                osn_tsdf_cuda_synchronize(rv);

                std::vector<float> rbuf((std::size_t)(4 << 20) * 6);
                const int32_t rn = osn_tsdf_cuda_extract_mesh(rv, rbuf.data(), nullptr, 4 << 20,
                                                              0.5f, 0.0f);
                const std::size_t r_nv = rn > 0 ? (std::size_t)rn : 0;
                rbuf.resize(r_nv * 6);
                std::printf("  A4: %d blocks, %zu vertices\n", osn_tsdf_cuda_block_count(rv), r_nv);

                check("A4 produced geometry", r_nv > 1000, fmt("%.0f vertices", (double)r_nv));
                check("A4 no points dropped", osn_tsdf_cuda_drop_count(rv) == 0,
                      fmt("%.0f dropped", (double)osn_tsdf_cuda_drop_count(rv)));
                check("A3/A4 block counts identical",
                      osn_tsdf_cuda_block_count(rv) == gpu.block_count(),
                      fmt("%.0f vs %.0f", (double)osn_tsdf_cuda_block_count(rv),
                          (double)gpu.block_count()));

                double r_mean = 0.0, r_max = 0.0;
                for (std::size_t i = 0; i < r_nv; ++i) {
                    const double x = rbuf[i * 6], y = rbuf[i * 6 + 1], z = rbuf[i * 6 + 2];
                    const double e = std::fabs(std::sqrt(x * x + y * y + z * z) - R);
                    r_mean += e;
                    r_max = std::max(r_max, e);
                }
                if (r_nv) r_mean /= r_nv;
                check("A4 mean |r - R| < voxel/4", r_mean < voxel * 0.25,
                      fmt("%.6f m (voxel %.3f)", r_mean, voxel));
                check("A4 max |r - R| < voxel", r_max < voxel, fmt("%.6f m", r_max));

                if (r_nv && g_nv) {
                    const auto rpos = positions_of(rbuf);
                    const auto gpos2 = positions_of(gmesh);
                    GridNN gg(gpos2, voxel * 2.0f);
                    GridNN rr(rpos, voxel * 2.0f);
                    double s1 = 0.0, m1 = 0.0, s2 = 0.0, m2 = 0.0;
                    for (std::size_t i = 0; i < r_nv; i += kCrossCheckStride) {
                        const float d = gg.nearest(&rpos[i * 3]);
                        s1 += d; m1 = std::max(m1, (double)d);
                    }
                    for (std::size_t i = 0; i < g_nv; i += kCrossCheckStride) {
                        const float d = rr.nearest(&gpos2[i * 3]);
                        s2 += d; m2 = std::max(m2, (double)d);
                    }
                    const double nq = (double)((r_nv + kCrossCheckStride - 1) / kCrossCheckStride);
                    const double ng = (double)((g_nv + kCrossCheckStride - 1) / kCrossCheckStride);
                    const double mean_sym = 0.5 * (s1 / nq + s2 / ng);
                    const double haus = std::max(m1, m2);
                    std::printf("  A3 vs A4: mean %.9f m, hausdorff %.9f m\n", mean_sym, haus);
                    check("A3/A4 mean surface distance < voxel/100", mean_sym < voxel * 0.01,
                          fmt("%.9f m", mean_sym));
                    check("A3/A4 hausdorff < voxel/10", haus < voxel * 0.1, fmt("%.9f m", haus));
                }
            }
            if (rv) osn_tsdf_cuda_destroy(rv);
        }
    }
#endif

    // ---- A3 on the plane --------------------------------------------------
    cudaFree(d_pts);
    {
        const float z0 = 0.30f;
        const auto ppts = scenes::plane(z0, 0.25f, 500);
        float* d_pp = upload(ppts);
        CudaVolume pv(cfg);
        PointBatch pb;
        pb.positions = d_pp;
        pb.n = (int32_t)(ppts.size() / 3);
        pv.integrate(pb);
        pv.synchronize();
        const auto pmesh = extract_gpu(pv, 0.5f);
        const std::size_t nv = pmesh.size() / 6;
        double mean_dz = 0.0, max_dz = 0.0;
        for (std::size_t i = 0; i < nv; ++i) {
            const double dz = std::fabs(pmesh[i * 6 + 2] - z0);
            mean_dz += dz;
            max_dz = std::max(max_dz, dz);
        }
        if (nv) mean_dz /= nv;
        std::printf("\n--- A3 plane z=%.2f m ---\n  vertices %zu\n", z0, nv);
        check("A3 plane produced geometry", nv > 1000, fmt("%.0f vertices", (double)nv));
        check("A3 mean |z - z0| < voxel/4", mean_dz < voxel * 0.25, fmt("%.6f m", mean_dz));
        check("A3 max |z - z0| < voxel", max_dz < voxel, fmt("%.6f m", max_dz));
        cudaFree(d_pp);
    }

    std::printf("\n=== %s ===\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
