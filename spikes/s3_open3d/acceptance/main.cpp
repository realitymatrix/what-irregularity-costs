// S3 acceptance test for the Open3D (A1) arm.
//
// Three questions, not one:
//   1. Does VoxelBlockGrid work from C++ on CPU?
//   2. Does it work on CUDA (sm_120)? This is what the upstream stdgpu
//      backport unblocks.
//   3. Is the mesh-comparison API present? The Phase 1 correctness gate needs
//      it regardless of whether A1 ever ships, so it is tested separately.
//
// The CPU/CUDA cross-check is deliberate: if the two devices disagree on block
// count for identical input, A1 cannot serve as a baseline no matter which one
// is "right".
//
// Build:
//   cmake .. -DOpen3D_ROOT=$HOME/.local/opt/open3d -DCMAKE_BUILD_TYPE=Release
#include <open3d/Open3D.h>

#include <cstdio>
#include <utility>
#include <vector>

using namespace open3d;

namespace {

struct VbgResult {
    int64_t blocks = 0;
    size_t vertices = 0;
    size_t triangles = 0;
    bool ran = false;
};

// Synthetic depth: a plane at 2 m, at TartanAir V2 intrinsics
// (640x640, fx = fy = 320, cx = cy = 320).
VbgResult RunVoxelBlockGrid(const core::Device& dev) {
    VbgResult r;
    t::geometry::VoxelBlockGrid vbg(
            {"tsdf", "weight", "color"},
            {core::Float32, core::Float32, core::Float32},
            {{1}, {1}, {3}},
            3.0f / 512.0f, 16, 1000, dev);

    const int W = 640, H = 640;
    core::Tensor depth = core::Tensor::Full({H, W, 1}, 2000.0f, core::Float32, dev);
    t::geometry::Image depth_img(depth);
    core::Tensor intrinsic = core::Tensor::Init<double>(
            {{320.0, 0.0, 320.0}, {0.0, 320.0, 320.0}, {0.0, 0.0, 1.0}});
    core::Tensor extrinsic =
            core::Tensor::Eye(4, core::Float64, core::Device("CPU:0"));

    core::Tensor frustum = vbg.GetUniqueBlockCoordinates(
            depth_img, intrinsic, extrinsic, 1000.0f, 5.0f);

    // ExtractTriangleMesh defaults to weight_threshold = 3.0, so a single
    // integration extracts zero vertices and looks exactly like a broken TSDF.
    // Integrate repeatedly and lower the threshold.
    for (int i = 0; i < 4; ++i) {
        vbg.Integrate(frustum, depth_img, intrinsic, extrinsic, 1000.0f, 5.0f);
    }

    auto legacy = vbg.ExtractTriangleMesh(0.5f).ToLegacy();
    r.blocks = vbg.GetHashMap().Size();
    r.vertices = legacy.vertices_.size();
    r.triangles = legacy.triangles_.size();
    r.ran = true;
    return r;
}

bool MeshComparisonWorks() {
    // The Phase 1 gate needs surface-to-surface and Hausdorff distance between
    // meshes. Verify the API exists and is monotone in a known perturbation.
    auto a = geometry::TriangleMesh::CreateSphere(1.0, 40);
    auto near_m = geometry::TriangleMesh::CreateSphere(1.01, 40);
    auto far_m = geometry::TriangleMesh::CreateSphere(1.10, 40);

    auto pa = a->SamplePointsUniformly(20000);
    auto pn = near_m->SamplePointsUniformly(20000);
    auto pf = far_m->SamplePointsUniformly(20000);

    auto stats = [](const std::vector<double>& d) {
        double mean = 0.0, haus = 0.0;
        for (double v : d) {
            mean += v;
            if (v > haus) haus = v;
        }
        return std::pair<double, double>{mean / d.size(), haus};
    };

    auto near_s = stats(pa->ComputePointCloudDistance(*pn));
    auto far_s = stats(pa->ComputePointCloudDistance(*pf));

    printf("  r=1.00 vs r=1.01 : mean %.4f  hausdorff %.4f\n",
           near_s.first, near_s.second);
    printf("  r=1.00 vs r=1.10 : mean %.4f  hausdorff %.4f\n",
           far_s.first, far_s.second);

    // Monotonicity is the real property the gate depends on. Absolute values
    // are dominated by point-sampling spacing, not the radius difference, so
    // do not assert on them.
    return far_s.first > near_s.first && far_s.second > near_s.second;
}

}  // namespace

int main() {
    printf("Open3D %s\n", OPEN3D_VERSION);
    const bool cuda_available = core::Device("CUDA:0").IsAvailable();
    printf("CUDA available: %s\n\n", cuda_available ? "yes" : "no");

    printf("--- VoxelBlockGrid (A1 arm) ---\n");
    VbgResult cpu = RunVoxelBlockGrid(core::Device("CPU:0"));
    printf("  CPU:0   blocks %5ld   vertices %7zu   triangles %7zu\n",
           (long)cpu.blocks, cpu.vertices, cpu.triangles);

    VbgResult gpu;
    if (cuda_available) {
        gpu = RunVoxelBlockGrid(core::Device("CUDA:0"));
        printf("  CUDA:0  blocks %5ld   vertices %7zu   triangles %7zu\n",
               (long)gpu.blocks, gpu.vertices, gpu.triangles);
    } else {
        printf("  CUDA:0  SKIPPED (no CUDA device)\n");
    }

    bool cpu_ok = cpu.blocks > 0 && cpu.vertices > 0;
    bool gpu_ok = cuda_available && gpu.ran && gpu.blocks > 0 && gpu.vertices > 0;
    bool agree = cuda_available && gpu.ran && gpu.blocks == cpu.blocks;

    printf("\n--- Mesh comparison (Phase 1 correctness gate) ---\n");
    bool cmp_ok = MeshComparisonWorks();

    printf("\n%-34s %s\n", "VoxelBlockGrid CPU", cpu_ok ? "PASS" : "FAIL");
    printf("%-34s %s\n", "VoxelBlockGrid CUDA",
           !cuda_available ? "SKIP" : (gpu_ok ? "PASS" : "FAIL"));
    printf("%-34s %s\n", "CPU/CUDA block-count agreement",
           !cuda_available ? "SKIP" : (agree ? "PASS" : "FAIL"));
    printf("%-34s %s\n", "Mesh comparison (P1 gate)", cmp_ok ? "PASS" : "FAIL");

    const bool ok = cpu_ok && cmp_ok && (!cuda_available || (gpu_ok && agree));
    printf("\n=== %s ===\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
