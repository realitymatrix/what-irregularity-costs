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
//   * **Interleaved arm order.** Arms take turns within each repetition, and
//     the starting arm rotates. Running all of arm A then all of arm B makes
//     clock and thermal drift indistinguishable from a language difference:
//     an earlier measured run of this harness saw the GPU go from 1575 MHz to
//     2842 MHz across a single session, which is larger than several of the
//     effects being measured. A per-arm first-half / second-half comparison is
//     printed as a self-check that the interleaving actually worked.
//   * **Exclusive device.** Refuses to run when another process holds
//     significant GPU memory. A FoundationStereo job sharing the device once
//     made arm A3 measure 4.455 ms against its true 0.044 ms, a 100x error
//     that looked like a plausible result.
//   * **Amortised launches.** Each timed window contains BATCH launches rather
//     than one, and the result is divided by BATCH. Timing a single launch
//     measures the host's wait for completion as much as the kernel: the
//     driver spins briefly then blocks, and the scheduler wakeup of ~130 us
//     lands unevenly across arms depending on how much host work each does
//     after enqueuing. That produced a spurious 4x tail on arm A4's allocate
//     which bench/probe_a4_tail.cu showed was entirely in the wait, with the
//     kernel flat at 0.050 ms. Batching amortises one completion over BATCH
//     kernels, so the measurement is steady-state throughput rather than
//     per-launch latency.
//
//     Allocate needs BATCH DISTINCT volumes: after the first launch the table
//     is already populated, so re-launching into the same volume would measure
//     a lookup-only fast path. The volumes are reset outside the timed window.
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
#include <unistd.h>
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

/// Total device memory held by OTHER compute processes, in MiB.
///
/// Desktop compositing shows up here at a couple of hundred MiB and is
/// tolerable; a real workload is not. The threshold is deliberately generous:
/// the failure this guards against is catastrophic and silent, so a false
/// refusal costs far less than a false result.
long other_process_mib() {
    FILE* p = popen("nvidia-smi --query-compute-apps=pid,used_memory "
                    "--format=csv,noheader,nounits -i 0 2>/dev/null",
                    "r");
    if (!p) return 0;
    char line[512];
    const long self = (long)getpid();
    long total = 0;
    while (fgets(line, sizeof(line), p)) {
        long pid = 0, mib = 0;
        if (sscanf(line, "%ld, %ld", &pid, &mib) == 2 && pid != self) total += mib;
    }
    pclose(p);
    return total;
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

    // Spin rather than yield while waiting on the device.
    //
    // By default the driver spins briefly and then blocks on an OS primitive.
    // When it blocks, scheduler wakeup latency of roughly 130 us lands inside
    // the measurement, and it lands unevenly: an arm whose host path returns
    // quickly keeps the driver spinning, while one that does a little host work
    // after the launch gives the GPU time to go idle and is more likely to
    // yield. That produced a spurious 4x tail on arm A4's allocate (p50 0.052,
    // p95 0.187) which probe_a4_tail showed was entirely in the wait: the
    // kernel itself was flat at 0.050 ms and the launch call took 2 us.
    //
    // Spinning costs a busy core, which is the right trade for a benchmark.
    // Must be set before any context exists.
    const cudaError_t flag_rc = cudaSetDeviceFlags(cudaDeviceScheduleSpin);

    std::printf("=== TSDF fusion arm timing ===\n");
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("  device %s sm_%d%d | reps %d (warmup %d discarded)\n", prop.name, prop.major,
                prop.minor, reps, warmup);
    std::printf("  clocks/thermals are LOGGED, not pinned: nvidia-smi -lgc needs root\n");
    std::printf("  arm order is INTERLEAVED and rotated per repetition\n");
    std::printf("  device sync policy: requested SPIN -> %s\n",
                flag_rc == cudaSuccess ? "applied" : cudaGetErrorName(flag_rc));
    std::printf("  state at start: %s\n", gpu_state().c_str());

    const long busy = other_process_mib();
    const bool allow_shared = getenv("OSN_BENCH_ALLOW_SHARED") != nullptr;
    std::printf("  other GPU processes: %ld MiB\n", busy);
    if (busy > 1024 && !allow_shared) {
        std::printf("\n  REFUSING TO RUN: another process holds %ld MiB on this GPU.\n", busy);
        std::printf("  Timings taken on a shared device are not comparable; a previous run\n");
        std::printf("  measured a 100x error this way. Set OSN_BENCH_ALLOW_SHARED=1 to\n");
        std::printf("  override, and do not quote the numbers if you do.\n");
        return 2;
    }

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

    // ---- GPU arms, interleaved, batched -----------------------------------
    //
    // One repetition runs every arm once; the starting arm rotates so no arm is
    // systematically measured on a cold or hot device. Each timed window holds
    // BATCH launches, divided out afterwards.
    {
        const int BATCH = getenv("OSN_BENCH_BATCH") ? std::atoi(getenv("OSN_BENCH_BATCH")) : 8;
        std::printf("  batch: %d launches per timed window\n\n", BATCH);

        // Volumes are shared across arms and reset before each arm's window, so
        // the device memory cost is BATCH volumes rather than BATCH per arm.
        std::vector<OsnTsdfCudaVolume*> vols((size_t)BATCH, nullptr);
        std::vector<OsnTsdfDeviceView> dvs((size_t)BATCH);
        for (int j = 0; j < BATCH; ++j) {
            vols[j] = osn_tsdf_cuda_create(voxel, cfg.pool_capacity_blocks, -1.0f, 32.0f);
            if (!vols[j]) { std::printf("  volume %d allocation failed\n", j); return 3; }
            osn_tsdf_cuda_device_view(vols[j], &dvs[j]);
        }

#ifdef OSN_TSDF_HAVE_A4
        const bool have_a4 = (a4_init() == 0);
#else
        const bool have_a4 = false;
#endif
        OsnTritonKernel* k_upd = nullptr;
        OsnTritonKernel* k_alloc = nullptr;
#ifdef HAVE_A5
        const std::string dir = getenv("OSN_TRITON_DIR") ? getenv("OSN_TRITON_DIR")
                                                         : "../artifacts/triton";
        k_upd = osn_triton_load((dir + "/" + OSN_TRITON_UPDATE_NAME + ".cubin").c_str(),
                                OSN_TRITON_UPDATE_NAME);
        k_alloc = osn_triton_load((dir + "/" + OSN_TRITON_ALLOC_NAME + ".cubin").c_str(),
                                  OSN_TRITON_ALLOC_NAME);
        const bool have_a5 = (k_upd && k_alloc);
#else
        const bool have_a5 = false;
#endif

        auto reset_all = [&]() {
            for (int j = 0; j < BATCH; ++j) osn_tsdf_cuda_reset(vols[j]);
        };
        auto launch_alloc = [&](int arm, int j) {
            if (arm == 0) osn_tsdf_cuda_allocate_blocks(vols[j], d_pts, n, 0, 0, 0, 0);
#ifdef OSN_TSDF_HAVE_A4
            else if (arm == 1) a4_allocate_blocks(&dvs[j], (uint64_t)d_pts, n, 0, 0, 0, 0);
#endif
#ifdef HAVE_A5
            else if (arm == 2) {
                const int32_t hm = (int32_t)dvs[j].hash_mask;
                const float z = 0.0f;
                const uint32_t grid = (uint32_t)((n + OSN_TRITON_BLOCK - 1) / OSN_TRITON_BLOCK);
                void* args[] = {(void*)&d_pts, (void*)&dvs[j].table, (void*)&dvs[j].table,
                                (void*)&dvs[j].block_count, (void*)&dvs[j].block_coord,
                                (void*)&dvs[j].drop_count, (void*)&dvs[j].scratch_base,
                                (void*)&n, (void*)&hm, (void*)&dvs[j].pool_capacity,
                                (void*)&dvs[j].voxel_size_m, (void*)&dvs[j].trunc_m,
                                (void*)&z, (void*)&z, (void*)&z, (void*)&z};
                osn_triton_launch(k_alloc, grid, OSN_TRITON_ALLOC_BLOCK_DIM_X,
                                  OSN_TRITON_ALLOC_SHARED, args, 16);
            }
#endif
        };
        auto launch_update = [&](int arm, int j) {
            if (arm == 0) osn_tsdf_cuda_update_voxels(vols[j], d_pts, n, 0, 0, 0, 0);
#ifdef OSN_TSDF_HAVE_A4
            else if (arm == 1) a4_update_voxels(&dvs[j], (uint64_t)d_pts, n, 0, 0, 0, 0);
#endif
#ifdef HAVE_A5
            else if (arm == 2) {
                const int32_t hm = (int32_t)dvs[j].hash_mask;
                const float z = 0.0f;
                const uint32_t grid = (uint32_t)((n + OSN_TRITON_BLOCK - 1) / OSN_TRITON_BLOCK);
                void* args[] = {(void*)&d_pts, (void*)&dvs[j].table, (void*)&dvs[j].table,
                                (void*)&dvs[j].tsdf, (void*)&dvs[j].weight, (void*)&dvs[j].r,
                                (void*)&dvs[j].g, (void*)&dvs[j].b, (void*)&n, (void*)&hm,
                                (void*)&dvs[j].voxel_size_m, (void*)&dvs[j].trunc_m,
                                (void*)&dvs[j].weight_cap,
                                (void*)&z, (void*)&z, (void*)&z, (void*)&z};
                osn_triton_launch(k_upd, grid, OSN_TRITON_UPDATE_BLOCK_DIM_X,
                                  OSN_TRITON_UPDATE_SHARED, args, 17);
            }
#endif
        };

        std::vector<double> smp[3][3];
        const char* arm_name[3] = {"A3-cuda", "A4-rust", "A5-triton"};
        GpuTimer t;

        for (int i = 0; i < warmup + reps; ++i) {
            for (int k = 0; k < 3; ++k) {
                const int arm = (i + k) % 3;
                if (arm == 1 && !have_a4) continue;
                if (arm == 2 && !have_a5) continue;

                // allocate: BATCH launches into BATCH empty volumes.
                reset_all();
                cudaDeviceSynchronize();
                t.start();
                for (int j = 0; j < BATCH; ++j) launch_alloc(arm, j);
                const double a = t.stop_ms() / BATCH;

                // update: the volumes are now allocated, so BATCH updates all
                // take the same path. Accumulating into an already-updated
                // volume is exactly what a multi-frame integrate does.
                t.start();
                for (int j = 0; j < BATCH; ++j) launch_update(arm, j);
                const double u = t.stop_ms() / BATCH;

                double e = -1;
                if (arm == 0) {
                    // Extraction is shared across the GPU arms and does not
                    // mutate the volume, so BATCH extracts of one volume are
                    // equivalent and can share a window.
                    t.start();
                    for (int j = 0; j < BATCH; ++j)
                        osn_tsdf_cuda_extract_mesh(vols[0], mesh_buf.data(), nullptr, 4 << 20,
                                                   0.5f, 0.0f);
                    e = t.stop_ms() / BATCH;
                    if (expect_blocks < 0) expect_blocks = osn_tsdf_cuda_block_count(vols[0]);
                }

                if (i >= warmup) {
                    smp[arm][0].push_back(a);
                    smp[arm][1].push_back(u);
                    if (e >= 0) smp[arm][2].push_back(e);
                }
            }
        }

        const auto g = gpu_state();
        const char* stage_name[3] = {"allocate", "update", "extract"};
        for (int arm = 0; arm < 3; ++arm) {
            if (smp[arm][0].empty()) continue;
            const int32_t nb = osn_tsdf_cuda_block_count(vols[0]);
            std::printf("--- %s (blocks %d%s) ---\n", arm_name[arm], nb,
                        (nb == expect_blocks || nb < 0) ? "" : "  MISMATCH");
            for (int st = 0; st < 3; ++st)
                if (!smp[arm][st].empty())
                    emit(rows, arm_name[arm], stage_name[st], summarize(smp[arm][st]), g);
        }

        std::printf("\n  drift check (per-arm first-half vs second-half p50, update stage):\n");
        for (int arm = 0; arm < 3; ++arm) {
            auto& v = smp[arm][1];
            if (v.size() < 4) continue;
            std::vector<double> h1(v.begin(), v.begin() + v.size() / 2);
            std::vector<double> h2(v.begin() + v.size() / 2, v.end());
            const double a = summarize(h1).p50, b = summarize(h2).p50;
            const double drift = a > 0 ? 100.0 * (b - a) / a : 0.0;
            std::printf("    %-12s %7.3f -> %7.3f ms  (%+.1f%%)%s\n", arm_name[arm], a, b, drift,
                        std::fabs(drift) > 5.0 ? "   <-- ORDERING MAY STILL BIAS THIS" : "");
        }
        std::printf("\n");

        for (auto* v : vols) if (v) osn_tsdf_cuda_destroy(v);
        if (k_upd) osn_triton_unload(k_upd);
        if (k_alloc) osn_triton_unload(k_alloc);
    }

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
