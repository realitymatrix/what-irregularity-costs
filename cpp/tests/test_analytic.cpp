// Tier-1 correctness: the CPU arm against closed-form geometry.
//
// This is the tier that makes the project's correctness argument work. No
// implementation, ours or anyone else's, defines what "correct" means here:
// the surface is known analytically, so the arm is measured against
// mathematics rather than against another program.
//
// Two scenes:
//   * Sphere. Every extracted vertex must lie within discretisation error of
//     radius R from the centre. Catches sign errors, scale errors, and any
//     confusion between voxel and world coordinates in one number.
//   * Plane. Every vertex must lie on z = z0. Catches the axis-ordering
//     mistakes a sphere is symmetric enough to hide.
//
// Also checks the properties a benchmark harness depends on but a geometric
// test would not think to assert: thread-count invariance (the result must not
// depend on how many workers ran) and pool-exhaustion reporting.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "osn_tsdf/volume.hpp"

using namespace osn_tsdf;

namespace {

int g_failures = 0;

void check(const char* label, bool pass, const std::string& detail) {
    if (!pass) ++g_failures;
    std::printf("  [%s] %s  %s\n", pass ? "PASS" : "FAIL", label, detail.c_str());
}

std::string fmt(const char* f, double a, double b = 0.0, double c = 0.0) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), f, a, b, c);
    return buf;
}

/// Points on a sphere, as if seen from a camera at the centre looking outward.
/// Viewing from inside keeps every surface point visible from one viewpoint,
/// so the sign convention is exercised without needing multiple views.
std::vector<float> sphere_points(float radius, int n_theta, int n_phi) {
    std::vector<float> p;
    p.reserve(static_cast<std::size_t>(n_theta) * n_phi * 3);
    for (int i = 0; i < n_theta; ++i) {
        const float theta = static_cast<float>(M_PI) * (i + 0.5f) / n_theta;
        for (int j = 0; j < n_phi; ++j) {
            const float phi = 2.0f * static_cast<float>(M_PI) * j / n_phi;
            p.push_back(radius * std::sin(theta) * std::cos(phi));
            p.push_back(radius * std::sin(theta) * std::sin(phi));
            p.push_back(radius * std::cos(theta));
        }
    }
    return p;
}

/// A z = z0 plane patch, seen from a camera below it.
std::vector<float> plane_points(float z0, float half_extent, int n) {
    std::vector<float> p;
    p.reserve(static_cast<std::size_t>(n) * n * 3);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            p.push_back(-half_extent + 2.0f * half_extent * i / (n - 1));
            p.push_back(-half_extent + 2.0f * half_extent * j / (n - 1));
            p.push_back(z0);
        }
    }
    return p;
}

std::vector<float> extract(Volume& v, float min_weight, int32_t cap = 4 << 20) {
    std::vector<float> posnor(static_cast<std::size_t>(cap) * 6);
    MeshBuffers out;
    out.posnor = posnor.data();
    out.rgb = nullptr;
    out.capacity_vertices = cap;
    const int32_t n = v.extract_mesh(out, min_weight, 0.0f);
    if (n < 0) return {};
    posnor.resize(static_cast<std::size_t>(n) * 6);
    return posnor;
}

}  // namespace

int main() {
    std::printf("=== A2 (CPU C++) analytic correctness ===\n\n");

    // ---- Sphere -----------------------------------------------------------
    {
        const float R = 0.5f;
        const float voxel = 0.01f;
        VolumeConfig cfg;
        cfg.voxel_size_m = voxel;
        cfg.pool_capacity_blocks = 1 << 14;
        Volume vol(cfg);

        const auto pts = sphere_points(R, 400, 800);
        PointBatch b;
        b.positions = pts.data();
        b.n = static_cast<int32_t>(pts.size() / 3);
        b.cam[0] = b.cam[1] = b.cam[2] = 0.0f;  // camera at the centre
        vol.integrate(b);

        const auto mesh = extract(vol, 0.5f);
        const std::size_t nv = mesh.size() / 6;
        std::printf("--- sphere R=%.2f m, voxel %.3f m ---\n", R, voxel);
        std::printf("  blocks %d, dropped %llu, vertices %zu\n", vol.block_count(),
                    static_cast<unsigned long long>(vol.drop_count()), nv);

        check("sphere produced geometry", nv > 1000, fmt("%.0f vertices", (double)nv));
        check("no points dropped", vol.drop_count() == 0,
              fmt("%.0f dropped", (double)vol.drop_count()));

        if (nv > 0) {
            double max_err = 0.0, sum_err = 0.0;
            for (std::size_t i = 0; i < nv; ++i) {
                const double x = mesh[i * 6 + 0], y = mesh[i * 6 + 1], z = mesh[i * 6 + 2];
                const double err = std::fabs(std::sqrt(x * x + y * y + z * z) - R);
                max_err = std::max(max_err, err);
                sum_err += err;
            }
            const double mean_err = sum_err / nv;
            // One voxel is the discretisation floor; the mean should be far
            // inside it, and the max within it.
            check("mean |r - R| < voxel/4", mean_err < voxel * 0.25,
                  fmt("%.6f m (voxel %.3f)", mean_err, voxel));
            check("max |r - R| < voxel", max_err < voxel,
                  fmt("%.6f m (voxel %.3f)", max_err, voxel));
        }
    }

    // ---- Plane ------------------------------------------------------------
    {
        const float z0 = 0.30f;
        const float voxel = 0.01f;
        VolumeConfig cfg;
        cfg.voxel_size_m = voxel;
        cfg.pool_capacity_blocks = 1 << 14;
        Volume vol(cfg);

        const auto pts = plane_points(z0, 0.25f, 500);
        PointBatch b;
        b.positions = pts.data();
        b.n = static_cast<int32_t>(pts.size() / 3);
        b.cam[0] = 0.0f; b.cam[1] = 0.0f; b.cam[2] = 0.0f;  // below the plane
        vol.integrate(b);

        const auto mesh = extract(vol, 0.5f);
        const std::size_t nv = mesh.size() / 6;
        std::printf("\n--- plane z=%.2f m, voxel %.3f m ---\n", z0, voxel);
        std::printf("  blocks %d, vertices %zu\n", vol.block_count(), nv);

        check("plane produced geometry", nv > 1000, fmt("%.0f vertices", (double)nv));
        if (nv > 0) {
            double max_dz = 0.0, sum_dz = 0.0;
            for (std::size_t i = 0; i < nv; ++i) {
                const double dz = std::fabs(mesh[i * 6 + 2] - z0);
                max_dz = std::max(max_dz, dz);
                sum_dz += dz;
            }
            check("mean |z - z0| < voxel/4", sum_dz / nv < voxel * 0.25,
                  fmt("%.6f m", sum_dz / nv));
            check("max |z - z0| < voxel", max_dz < voxel, fmt("%.6f m", max_dz));
        }
    }

    // ---- Thread-count invariance -----------------------------------------
    //
    // The update is a weighted mean folded in under a compare-exchange, so the
    // arithmetic is order-dependent in the last ulp. Geometry must nonetheless
    // be invariant: a benchmark that changes its answer with thread count
    // cannot be compared against anything.
    {
        const float R = 0.4f;
        const float voxel = 0.02f;
        const auto pts = sphere_points(R, 200, 400);

        auto run = [&](int threads) {
            VolumeConfig cfg;
            cfg.voxel_size_m = voxel;
            cfg.pool_capacity_blocks = 1 << 14;
            auto vol = std::make_unique<Volume>(cfg);
            vol->set_thread_count(threads);
            PointBatch b;
            b.positions = pts.data();
            b.n = static_cast<int32_t>(pts.size() / 3);
            vol->integrate(b);
            const auto mesh = extract(*vol, 0.5f);
            double sum = 0.0;
            const std::size_t nv = mesh.size() / 6;
            for (std::size_t i = 0; i < nv; ++i) {
                const double x = mesh[i * 6], y = mesh[i * 6 + 1], z = mesh[i * 6 + 2];
                sum += std::sqrt(x * x + y * y + z * z);
            }
            return std::pair<int32_t, double>{vol->block_count(), nv ? sum / nv : 0.0};
        };

        const auto one = run(1);
        const auto many = run(16);
        std::printf("\n--- thread-count invariance ---\n");
        std::printf("  1 thread : %d blocks, mean radius %.9f\n", one.first, one.second);
        std::printf("  16 threads: %d blocks, mean radius %.9f\n", many.first, many.second);
        check("block count independent of thread count", one.first == many.first,
              fmt("%.0f vs %.0f", (double)one.first, (double)many.first));
        check("mean radius independent of thread count",
              std::fabs(one.second - many.second) < 1e-5,
              fmt("delta %.3e m", std::fabs(one.second - many.second)));
    }

    // ---- Pool exhaustion is reported, not silent -------------------------
    {
        VolumeConfig cfg;
        cfg.voxel_size_m = 0.005f;
        cfg.pool_capacity_blocks = 16;  // far too small on purpose
        Volume vol(cfg);
        const auto pts = sphere_points(0.5f, 200, 400);
        PointBatch b;
        b.positions = pts.data();
        b.n = static_cast<int32_t>(pts.size() / 3);
        vol.integrate(b);
        std::printf("\n--- pool exhaustion ---\n");
        check("saturated pool is reported", vol.drop_count() > 0,
              fmt("%.0f drops, %.0f blocks", (double)vol.drop_count(),
                  (double)vol.block_count()));
        check("block count respects capacity", vol.block_count() <= 16,
              fmt("%.0f <= 16", (double)vol.block_count()));
    }

    std::printf("\n=== %s ===\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
