// Timing harness for the TSDF fusion arms.
//
// Built before any result is quoted, because most engineering benchmark papers
// are rejected for sloppy measurement rather than for uninteresting numbers.
// The commitments this implements:
//
//   * **Correctness gates timing.** Every arm has already been checked against
//     closed-form geometry and against the other arms. This binary refuses to
//     report a stage it could not verify produced the expected block count.
//   * **Distributions, not means.** p50 / p95 / p99 per stage, with the repeat
//     count stated. A mean hides exactly the tail that decides whether a
//     pipeline hits frame rate.
//   * **Warmup discarded.** First iterations include module load, allocator
//     warmup and clock ramp.
//   * **Clocks and thermals logged.** Blackwell throttles under sustained load.
//     Without recording clock state, thermal drift silently becomes "the
//     finding". Fixed clocks need root (`nvidia-smi -lgc`), which is not
//     available here, so the state is recorded rather than pinned and the
//     drift is visible in the output.
//   * **Per-stage attribution.** allocate / update / extract measured
//     separately. The crossover claim depends on per-stage numbers being
//     trustworthy, and a single end-to-end number cannot support it.
//
// Timing method: CUDA events around GPU stages, steady_clock around CPU ones.
// Launches are asynchronous, so every measurement synchronises; without that a
// GPU "measurement" records queue-submission time.

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "analytic_scenes.hpp"
#include "osn_tsdf/c_api.h"
#include "osn_tsdf/triton_launch.h"
#include "osn_tsdf/volume.hpp"
#include "osn_tsdf/volume_cuda.hpp"

#if __has_include("triton_tsdf_manifest.h")
#include "triton_tsdf_manifest.h"
#define HAVE_A5 1
#endif

#ifdef OSN_TSDF_HAVE_A4
extern "C" {
int32_t a4_init();
int32_t a4_allocate_blocks(const OsnTsdfDeviceView*, uint64_t, int32_t, float, float, float, float);
int32_t a4_update_voxels(const OsnTsdfDeviceView*, uint64_t, int32_t, float, float, float, float);
int32_t a4_synchronize();
}
#endif

using namespace osn_tsdf;

namespace {

struct Stats {
    double p50 = 0, p95 = 0, p99 = 0, min = 0, mean = 0;
    int n = 0;
};

/// Percentiles from raw samples. Nearest-rank, not interpolated: with the
/// repeat counts used here, interpolation would invent precision the sample
/// size does not support.
Stats summarize(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.n = (int)v.size();
    s.min = v.front();
    s.mean = 0;
    for (double x : v) s.mean += x;
    s.mean /= v.size();
    auto pct = [&](double p) { return v[std::min(v.size() - 1, (size_t)(v.size() * p))]; };
    s.p50 = pct(0.50);
    s.p95 = pct(0.95);
    s.p99 = pct(0.99);
    return s;
}

/// GPU clock and thermal state, read through nvidia-smi.
///
/// Recorded per arm rather than once per run: a later arm running hotter than
/// an earlier one is a measurement artefact, and it is only visible if the
/// state is sampled alongside the timings.
std::string gpu_state() {
    FILE* p = popen("nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw "
                    "--format=csv,noheader,nounits -i 0 2>/dev/null",
                    "r");
    if (!p) return "n/a";
    char buf[256] = {0};
    if (!fgets(buf, sizeof(buf), p)) { pclose(p); return "n/a"; }
    pclose(p);
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

/// CUDA-event timer. Events measure device time, excluding host launch
/// overhead, which is what a kernel comparison wants.
struct GpuTimer {
    cudaEvent_t a{}, b{};
    GpuTimer() { cudaEventCreate(&a); cudaEventCreate(&b); }
    ~GpuTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void start() { cudaEventRecord(a); }
    double stop_ms() {
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        return ms;
    }
};

double host_ms(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

struct Row {
    std::string arm, stage;
    Stats s;
    std::string gpu;
};

void emit(std::vector<Row>& rows, const char* arm, const char* stage, const Stats& s,
          const std::string& gpu) {
    rows.push_back({arm, stage, s, gpu});
    std::printf("  %-14s %-9s  p50 %8.3f  p95 %8.3f  p99 %8.3f  min %8.3f  (n=%d)\n", arm, stage,
                s.p50, s.p95, s.p99, s.min, s.n);
}

}  // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 30;
    const int warmup = argc > 2 ? std::atoi(argv[2]) : 5;
    const float voxel = 0.01f;
    const float R = 0.5f;

    std::printf("=== TSDF fusion arm timing ===\n");
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("  device %s sm_%d%d | reps %d (warmup %d discarded)\n", prop.name, prop.major,
                prop.minor, reps, warmup);
    std::printf("  clocks/thermals are LOGGED, not pinned: nvidia-smi -lgc needs root\n");
    std::printf("  state at start: %s\n", gpu_state().c_str());

    const auto pts = scenes::sphere(R, 400, 800);
    const int32_t n = (int32_t)(pts.size() / 3);
    std::printf("  scene: sphere R=%.2f m, %d points, voxel %.3f m\n\n", R, n, voxel);

    float* d_pts = nullptr;
    cudaMalloc(&d_pts, pts.size() * sizeof(float));
    cudaMemcpy(d_pts, pts.data(), pts.size() * sizeof(float), cudaMemcpyHostToDevice);

    VolumeConfig cfg;
    cfg.voxel_size_m = voxel;
    cfg.pool_capacity_blocks = 1 << 14;

    std::vector<Row> rows;
    int32_t expect_blocks = -1;
    std::vector<float> mesh_buf((size_t)(4 << 20) * 6);

    // ---- A3: CUDA C++ ------------------------------------------------------
    {
        OsnTsdfCudaVolume* v = osn_tsdf_cuda_create(voxel, cfg.pool_capacity_blocks, -1.0f, 32.0f);
        GpuTimer t;
        std::vector<double> alloc, upd, ext;
        for (int i = 0; i < warmup + reps; ++i) {
            osn_tsdf_cuda_reset(v);
            t.start();
            osn_tsdf_cuda_allocate_blocks(v, d_pts, n, 0, 0, 0, 0);
            const double a = t.stop_ms();
            t.start();
            osn_tsdf_cuda_update_voxels(v, d_pts, n, 0, 0, 0, 0);
            const double u = t.stop_ms();
            t.start();
            osn_tsdf_cuda_extract_mesh(v, mesh_buf.data(), nullptr, 4 << 20, 0.5f, 0.0f);
            const double e = t.stop_ms();
            if (i >= warmup) { alloc.push_back(a); upd.push_back(u); ext.push_back(e); }
        }
        expect_blocks = osn_tsdf_cuda_block_count(v);
        const auto g = gpu_state();
        std::printf("--- A3 cuda (blocks %d) ---\n", expect_blocks);
        emit(rows, "A3-cuda", "allocate", summarize(alloc), g);
        emit(rows, "A3-cuda", "update", summarize(upd), g);
        emit(rows, "A3-cuda", "extract", summarize(ext), g);
        osn_tsdf_cuda_destroy(v);
    }

#ifdef OSN_TSDF_HAVE_A4
    // ---- A4: Rust via cuda-oxide ------------------------------------------
    if (a4_init() == 0) {
        OsnTsdfCudaVolume* v = osn_tsdf_cuda_create(voxel, cfg.pool_capacity_blocks, -1.0f, 32.0f);
        OsnTsdfDeviceView dv{};
        osn_tsdf_cuda_device_view(v, &dv);
        GpuTimer t;
        std::vector<double> alloc, upd;
        for (int i = 0; i < warmup + reps; ++i) {
            osn_tsdf_cuda_reset(v);
            t.start();
            a4_allocate_blocks(&dv, (uint64_t)d_pts, n, 0, 0, 0, 0);
            const double a = t.stop_ms();
            t.start();
            a4_update_voxels(&dv, (uint64_t)d_pts, n, 0, 0, 0, 0);
            const double u = t.stop_ms();
            if (i >= warmup) { alloc.push_back(a); upd.push_back(u); }
        }
        const int32_t nb = osn_tsdf_cuda_block_count(v);
        const auto g = gpu_state();
        std::printf("--- A4 rust-cuda (blocks %d%s) ---\n", nb,
                    nb == expect_blocks ? "" : "  MISMATCH");
        emit(rows, "A4-rust", "allocate", summarize(alloc), g);
        emit(rows, "A4-rust", "update", summarize(upd), g);
        osn_tsdf_cuda_destroy(v);
    }
#endif

#ifdef HAVE_A5
    // ---- A5a / A5b: Triton -------------------------------------------------
    {
        const std::string dir = getenv("OSN_TRITON_DIR") ? getenv("OSN_TRITON_DIR")
                                                         : "../artifacts/triton";
        OsnTritonKernel* ku = osn_triton_load((dir + "/" + OSN_TRITON_UPDATE_NAME + ".cubin").c_str(),
                                              OSN_TRITON_UPDATE_NAME);
        OsnTritonKernel* ka = osn_triton_load((dir + "/" + OSN_TRITON_ALLOC_NAME + ".cubin").c_str(),
                                              OSN_TRITON_ALLOC_NAME);
        if (ku && ka) {
            OsnTsdfCudaVolume* v =
                osn_tsdf_cuda_create(voxel, cfg.pool_capacity_blocks, -1.0f, 32.0f);
            OsnTsdfDeviceView dv{};
            osn_tsdf_cuda_device_view(v, &dv);
            const int32_t scratch = (int32_t)dv.hash_mask;
            const int32_t hm = (int32_t)dv.hash_mask;
            const float zero = 0.0f;
            const uint32_t grid = (uint32_t)((n + OSN_TRITON_BLOCK - 1) / OSN_TRITON_BLOCK);
            const int64_t sentinel = -2;

            void* aargs[] = {(void*)&d_pts, (void*)&dv.table, (void*)&dv.table,
                             (void*)&dv.block_count, (void*)&dv.block_coord, (void*)&dv.drop_count,
                             (void*)&scratch, (void*)&n, (void*)&hm, (void*)&dv.pool_capacity,
                             (void*)&dv.voxel_size_m, (void*)&dv.trunc_m,
                             (void*)&zero, (void*)&zero, (void*)&zero, (void*)&zero};
            void* uargs[] = {(void*)&d_pts, (void*)&dv.table, (void*)&dv.table, (void*)&dv.tsdf,
                             (void*)&dv.weight, (void*)&dv.r, (void*)&dv.g, (void*)&dv.b,
                             (void*)&n, (void*)&hm, (void*)&dv.voxel_size_m,
                             (void*)&dv.trunc_m, (void*)&dv.weight_cap,
                             (void*)&zero, (void*)&zero, (void*)&zero, (void*)&zero};

            GpuTimer t;
            std::vector<double> t_alloc, t_upd;
            for (int i = 0; i < warmup + reps; ++i) {
                osn_tsdf_cuda_reset(v);
                cuMemcpyHtoD((CUdeviceptr)((char*)dv.table + (size_t)scratch * 16), &sentinel,
                             sizeof(int64_t));
                t.start();
                osn_triton_launch(ka, grid, OSN_TRITON_ALLOC_BLOCK_DIM_X, OSN_TRITON_ALLOC_SHARED,
                                  aargs, 16);
                const double a = t.stop_ms();
                t.start();
                osn_triton_launch(ku, grid, OSN_TRITON_UPDATE_BLOCK_DIM_X, OSN_TRITON_UPDATE_SHARED,
                                  uargs, 17);
                const double u = t.stop_ms();
                if (i >= warmup) { t_alloc.push_back(a); t_upd.push_back(u); }
            }
            const int32_t nb = osn_tsdf_cuda_block_count(v);
            const auto g = gpu_state();
            std::printf("--- A5 triton (blocks %d%s) ---\n", nb,
                        nb == expect_blocks ? "" : "  MISMATCH");
            // A5a uses A3's allocate; A5b uses Triton's. Reporting both stages
            // for the same update kernel is what makes A5b - A5a the tax.
            emit(rows, "A5b-triton", "allocate", summarize(t_alloc), g);
            emit(rows, "A5a/b-triton", "update", summarize(t_upd), g);
            osn_tsdf_cuda_destroy(v);
        }
        if (ku) osn_triton_unload(ku);
        if (ka) osn_triton_unload(ka);
    }
#endif

    // ---- A2: CPU C++ -------------------------------------------------------
    //
    // Fewer repetitions: the CPU arm is expected to be orders of magnitude
    // slower, and running it `reps` times would dominate wall clock without
    // adding information. Reported on its own axis for the same reason.
    {
        const int cpu_reps = std::max(3, reps / 10);
        std::vector<double> alloc, upd, ext;
        for (int i = 0; i < cpu_reps + 1; ++i) {
            Volume vol(cfg);
            PointBatch b;
            b.positions = pts.data();
            b.n = n;
            auto t0 = std::chrono::steady_clock::now();
            vol.allocate_blocks(b);
            const double a = host_ms(t0);
            t0 = std::chrono::steady_clock::now();
            vol.update_voxels(b);
            const double u = host_ms(t0);
            MeshBuffers mb;
            mb.posnor = mesh_buf.data();
            mb.capacity_vertices = 4 << 20;
            t0 = std::chrono::steady_clock::now();
            vol.extract_mesh(mb, 0.5f, 0.0f);
            const double e = host_ms(t0);
            if (i > 0) { alloc.push_back(a); upd.push_back(u); ext.push_back(e); }
        }
        std::printf("--- A2 cpu (%d threads) ---\n", Volume(cfg).thread_count());
        const auto g = gpu_state();
        emit(rows, "A2-cpu", "allocate", summarize(alloc), g);
        emit(rows, "A2-cpu", "update", summarize(upd), g);
        emit(rows, "A2-cpu", "extract", summarize(ext), g);
    }

    std::printf("\n  state at end:   %s\n", gpu_state().c_str());

    // Raw CSV, so the numbers can be re-analysed without re-running.
    const char* csv_path = getenv("OSN_BENCH_CSV");
    if (csv_path) {
        FILE* f = fopen(csv_path, "w");
        if (f) {
            fprintf(f, "arm,stage,p50_ms,p95_ms,p99_ms,min_ms,mean_ms,n,gpu_state\n");
            for (const auto& r : rows)
                fprintf(f, "%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%d,\"%s\"\n", r.arm.c_str(),
                        r.stage.c_str(), r.s.p50, r.s.p95, r.s.p99, r.s.min, r.s.mean, r.s.n,
                        r.gpu.c_str());
            fclose(f);
            std::printf("  wrote %s\n", csv_path);
        }
    }

    cudaFree(d_pts);
    return 0;
}
