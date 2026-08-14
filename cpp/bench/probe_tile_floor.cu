// What does a tile model cost, independent of which tile language it is?
//
// A LinkedIn reader asked whether Triton's gap on this workload is a tile
// model consequence, and whether a different tile language such as cuTile
// would land closer to CUDA C++. That is two questions. The second needs
// cuTile. The first does not, because the tile model's defining constraint is
// stateable as arithmetic on probe depths:
//
//   per-thread model  a lane that resolves retires, so the launch costs the
//                     SUM of the lanes' probe depths
//   tile model        no lane retires until the program is done, so the launch
//                     costs the program's DEEPEST lane, charged to every lane
//   Triton today      neither: the trip count is a compile-time constant, so
//                     the launch costs that bound charged to every lane
//
// The first two are measured here from a single instrumented run. The third is
// arithmetic on the bound. The three together bound where any tile language
// can land: no implementation retires lanes early, so none goes below the
// second number, and a language with a dynamic block-uniform trip count and a
// maskable compare-exchange should approach it.
//
// This is a model, not a measurement of cuTile. It assumes a tile
// implementation aggregates perfectly and pays nothing else, which makes the
// number it produces a floor rather than a prediction. A real arm can only do
// worse than its own floor, so if cuTile lands below this, the model is wrong
// and that is worth knowing.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "analytic_scenes.hpp"
#include "osn_tsdf/types.hpp"
#include "osn_tsdf/volume_cuda.hpp"

namespace {

// Arm A5's compile-time probe bound, from the Triton source.
constexpr uint32_t kTritonBound = 8;
constexpr int kProgramThreads = 256;

struct Scene {
    std::string name;
    std::vector<float> pts;
    float cam[3] = {0, 0, 0};
    int32_t pool = 1 << 14;
};

bool read_osnp(const char* path, Scene& out) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    char magic[4];
    int32_t version = 0, n = 0;
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "OSNP", 4) != 0) {
        std::fclose(f);
        return false;
    }
    std::fread(&version, 4, 1, f);
    std::fread(&n, 4, 1, f);
    std::fread(out.cam, 4, 3, f);
    out.pts.resize((size_t)n * 3);
    std::fread(out.pts.data(), 4, out.pts.size(), f);
    std::fclose(f);
    return n > 0;
}

}  // namespace

int main(int argc, char** argv) {
    const float voxel = 0.01f;
    cudaSetDevice(0);

    std::vector<Scene> scenes;
    {
        Scene s;
        s.name = "sphere 320k (base)";
        s.pts = scenes::sphere(0.5f, 400, 800);
        scenes.push_back(s);
    }
    {
        Scene s;
        s.name = "sphere 320k, r=2.0 (r-2.0)";
        s.pts = scenes::sphere(2.0f, 400, 800);
        s.pool = 1 << 17;
        scenes.push_back(s);
    }
    for (int i = 1; i < argc; ++i) {
        Scene s;
        s.name = argv[i];
        s.pool = 1 << 16;
        if (read_osnp(argv[i], s)) scenes.push_back(s);
        else std::printf("  skipping %s (unreadable)\n", argv[i]);
    }

    std::printf("=== what a tile model costs, priced on probe depth ===\n");
    std::printf("  program = %d threads, Triton's compile-time bound = %u\n\n",
                kProgramThreads, kTritonBound);
    std::printf("  %-30s %12s %12s %12s\n", "scene", "per-thread", "tile floor", "ratio");
    std::printf("  ---------------------------------------------------------------------\n");

    for (auto& sc : scenes) {
        const int32_t n = (int32_t)(sc.pts.size() / 3);
        float* d_pts = nullptr;
        cudaMalloc(&d_pts, sc.pts.size() * sizeof(float));
        cudaMemcpy(d_pts, sc.pts.data(), sc.pts.size() * sizeof(float), cudaMemcpyHostToDevice);

        osn_tsdf::VolumeConfig cfg{};
        cfg.voxel_size_m = voxel;
        cfg.pool_capacity_blocks = sc.pool;
        osn_tsdf::CudaVolume vol(cfg);
        if (!vol.valid()) {
            std::printf("  %-30s volume allocation failed\n", sc.name.c_str());
            cudaFree(d_pts);
            continue;
        }
        osn_tsdf::PointBatch b{};
        b.positions = d_pts;
        b.n = n;
        b.cam[0] = sc.cam[0];
        b.cam[1] = sc.cam[1];
        b.cam[2] = sc.cam[2];
        b.radius_m = 0.0f;

        uint64_t lane = 0, tile = 0;
        vol.allocate_blocks_probe_depth(b, &lane, &tile);

        const double ratio = lane ? (double)tile / (double)lane : 0.0;
        std::printf("  %-30s %12llu %12llu %11.2fx\n", sc.name.c_str(),
                    (unsigned long long)lane, (unsigned long long)tile, ratio);

        cudaFree(d_pts);
    }

    std::printf("\n  per-thread  probe steps a model that retires lanes early runs.\n");
    std::printf("  tile floor  the same run charged at the deepest lane in each program.\n");
    std::printf("\n  The ratio is what a tile model pays before it does anything wrong: no\n");
    std::printf("  bad bound, no unmaskable exchange, perfect aggregation. It is a floor,\n");
    std::printf("  not a prediction, and an arm can only land above its own floor.\n");
    return 0;
}
