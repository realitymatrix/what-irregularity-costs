// S3 acceptance test. Two questions, not one:
//   1. Does VoxelBlockGrid (the A1 arm) work from C++ on CPU?
//   2. Is the mesh-comparison API present? Phase 1's correctness gate needs it
//      regardless of whether A1 ever ships.
#include <open3d/Open3D.h>
#include <cstdio>

using namespace open3d;

int main() {
    printf("Open3D %s\n", OPEN3D_VERSION);
    core::Device dev("CPU:0");

    // -- 1. VoxelBlockGrid, the A1 arm ---------------------------------
    t::geometry::VoxelBlockGrid vbg(
        {"tsdf", "weight", "color"},
        {core::Float32, core::Float32, core::Float32},
        {{1}, {1}, {3}},
        3.0f / 512.0f, 16, 1000, dev);

    // Synthetic 640x640 depth: a plane at 2 m, matching TartanAir intrinsics.
    const int W = 640, H = 640;
    core::Tensor depth = core::Tensor::Full({H, W, 1}, 2000.0f, core::Float32, dev);
    t::geometry::Image depth_img(depth);
    core::Tensor intrinsic = core::Tensor::Init<double>(
        {{320.0, 0.0, 320.0}, {0.0, 320.0, 320.0}, {0.0, 0.0, 1.0}});
    core::Tensor extrinsic = core::Tensor::Eye(4, core::Float64, core::Device("CPU:0"));

    core::Tensor frustum = vbg.GetUniqueBlockCoordinates(
        depth_img, intrinsic, extrinsic, 1000.0f, 5.0f);
    printf("  unique block coords: %ld\n", (long)frustum.GetLength());

    vbg.Integrate(frustum, depth_img, intrinsic, extrinsic, 1000.0f, 5.0f);
    printf("  hashmap size after integrate: %ld\n", (long)vbg.GetHashMap().Size());

    // Integrate the same frame a few times: ExtractTriangleMesh defaults to
    // weight_threshold=3.0, so a single observation is filtered out.
    for (int i = 0; i < 3; ++i) {
        vbg.Integrate(frustum, depth_img, intrinsic, extrinsic, 1000.0f, 5.0f);
    }
    auto mesh = vbg.ExtractTriangleMesh(0.5f);
    auto legacy = mesh.ToLegacy();
    printf("  mesh: %zu vertices, %zu triangles\n",
           legacy.vertices_.size(), legacy.triangles_.size());
    bool vbg_ok = vbg.GetHashMap().Size() > 0 && legacy.vertices_.size() > 0;

    // -- 2. Mesh comparison, what the Phase 1 gate needs ----------------
    auto a = geometry::TriangleMesh::CreateSphere(1.0);
    auto b = geometry::TriangleMesh::CreateSphere(1.02);
    auto pa = a->SamplePointsUniformly(2000);
    auto pb = b->SamplePointsUniformly(2000);
    std::vector<double> d = pa->ComputePointCloudDistance(*pb);
    double mean = 0.0, hausdorff = 0.0;
    for (double v : d) { mean += v; if (v > hausdorff) hausdorff = v; }
    mean /= d.size();
    printf("  sphere r=1.00 vs r=1.02: mean surface dist %.4f, hausdorff %.4f\n",
           mean, hausdorff);
    bool cmp_ok = mean > 0.005 && mean < 0.05;

    printf("\nVoxelBlockGrid (A1)        : %s\n", vbg_ok ? "PASS" : "FAIL");
    printf("Mesh comparison (P1 gate)  : %s\n", cmp_ok ? "PASS" : "FAIL");
    return (vbg_ok && cmp_ok) ? 0 : 1;
}
