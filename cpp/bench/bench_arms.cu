// Timing harness for the TSDF fusion arms, swept over a workload matrix.
//
// Built before any result is quoted, because most engineering benchmark papers
// are rejected for sloppy measurement rather than for uninteresting numbers.
// The commitments this implements:
//
//   * **Correctness gates timing.** Every arm has already been checked against
//     closed-form geometry and against the other arms. This binary refuses to
//     record a cell whose block count disagrees across arms, or whose pool
//     overflowed.
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
//     separately. A single end-to-end number cannot support a claim about
//     which language costs what, because the two stages differ in kind: one is
//     irregular and CAS-bound, the other regular and atomics-bound.
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
//
// ---------------------------------------------------------------------------
// The workload matrix, added 2026-07-30
// ---------------------------------------------------------------------------
//
// Until now every number came from one scene: a sphere at 320k points. That is
// not enough to call a ratio a property of a *language*, because it could just
// as easily be a property of that one scene's contention pattern. The sweep
// varies four things, each chosen to test a specific part of the explanation
// rather than to fill out a table:
//
//   1. **Point count at fixed geometry.** More points onto the same blocks
//      raises contention without changing the hash workload. Arms bound by
//      atomics should scale differently from arms bound by issue rate.
//   2. **Extent at fixed angular resolution.** A larger sphere spreads the same
//      points over more blocks, raising hash pressure while *lowering*
//      per-voxel contention. This is the opposite lever to (1), and the two
//      together separate hash cost from accumulation cost.
//   3. **Hash load factor.** The sharpest test of the Triton result. Triton's
//      allocate has no per-lane early exit, so it pays the full probe bound
//      regardless of load; CUDA C++ exits as soon as a lane resolves, so it
//      pays actual probe depth, which rises with load. The prediction is
//      therefore that the gap NARROWS as load factor rises. If it does not,
//      the published explanation is incomplete and should be weakened.
//   4. **Scene shape.** Sphere versus plane: a plane's blocks are contiguous
//      and its hash slots cluster differently.
//
// Load-factor cells size the pool from a measured block count rather than a
// guess, so the achieved load factor is reported rather than assumed.
//
// **Device axis.** One device per process, selected with --device. Both the
// Triton cubin and the cuda-oxide module bind to the context current at load
// time, so switching devices mid-run would need a reload per arm per device;
// one process per device removes the failure mode entirely. tools/sweep.sh
// drives the loop and accumulates one CSV.
//
// Note that the two local cards are the SAME architecture (both sm_120), so
// this axis varies machine width and bandwidth at FIXED codegen. It is not the
// second-architecture pass and must not be written up as one.

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// ---------------------------------------------------------------------------
// Workload description
// ---------------------------------------------------------------------------

enum class Shape { Sphere, Plane };

/// One cell of the sweep.
///
/// `pool_blocks == 0` means "size the pool from the measured block count using
/// `pool_factor`", which is how the load-factor axis works: the table is
/// next_pow2(pool * 2), so a pool of blocks*f targets a load factor near
/// 1/(2f) without anyone having to predict the block count in advance.
struct Cell {
    const char* name;
    const char* axis;  // which question this cell exists to answer
    Shape shape;
    float extent;      // sphere radius, or plane half-extent, in metres
    int a, b;          // sphere: n_theta, n_phi. plane: n, unused.
    int32_t pool_blocks;
    float pool_factor;
};

std::vector<float> make_points(const Cell& c) {
    if (c.shape == Shape::Sphere) return scenes::sphere(c.extent, c.a, c.b);
    return scenes::plane(0.5f, c.extent, c.a);
}

const char* shape_name(Shape s) { return s == Shape::Sphere ? "sphere" : "plane"; }

/// The default matrix.
///
/// Deliberately an explicit list rather than a cross product. A full cross
/// product of four axes would be mostly uninformative cells whose main effect
/// is to make the runtime long enough that nobody reruns the sweep. Each entry
/// below exists to answer a stated question.
const std::vector<Cell> kDefaultCells = {
    // The historical baseline, kept identical so every previously published
    // number stays comparable to the sweep.
    {"base",       "baseline",   Shape::Sphere, 0.5f,  400,  800, 1 << 14, 0},

    // 1. Point count at fixed geometry: same blocks, rising contention.
    {"pts-20k",    "points",     Shape::Sphere, 0.5f,  100,  200, 1 << 14, 0},
    {"pts-80k",    "points",     Shape::Sphere, 0.5f,  200,  400, 1 << 14, 0},
    {"pts-720k",   "points",     Shape::Sphere, 0.5f,  600, 1200, 1 << 14, 0},
    {"pts-1280k",  "points",     Shape::Sphere, 0.5f,  800, 1600, 1 << 14, 0},

    // 2. Extent at fixed angular resolution: same points, more blocks, less
    //    per-voxel contention. The opposite lever to axis 1.
    {"r-0.25",     "extent",     Shape::Sphere, 0.25f, 400,  800, 1 << 14, 0},
    {"r-1.0",      "extent",     Shape::Sphere, 1.0f,  400,  800, 1 << 16, 0},
    {"r-2.0",      "extent",     Shape::Sphere, 2.0f,  400,  800, 1 << 17, 0},

    // 3. Hash load factor, pool sized from the measured block count.
    //    Targets roughly 0.48, 0.25, 0.12, 0.03.
    {"lf-hi",      "loadfactor", Shape::Sphere, 0.5f,  400,  800, 0, 1.05f},
    {"lf-mid",     "loadfactor", Shape::Sphere, 0.5f,  400,  800, 0, 2.0f},
    {"lf-lo",      "loadfactor", Shape::Sphere, 0.5f,  400,  800, 0, 4.0f},
    {"lf-sparse",  "loadfactor", Shape::Sphere, 0.5f,  400,  800, 0, 16.0f},

    // 4. Shape. 566^2 = 320k points, matching the baseline's count, so the
    //    comparison is shape and not size.
    {"plane-320k", "shape",      Shape::Plane,  1.0f,  566,    0, 1 << 14, 0},
};

// ---------------------------------------------------------------------------
// Statistics and device state
// ---------------------------------------------------------------------------

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

/// CUDA ordinal used for cudaSetDevice.
int g_device = 0;
/// PHYSICAL index used for nvidia-smi and for the CSV.
///
/// These differ whenever the process is masked with CUDA_VISIBLE_DEVICES, which
/// is how tools/sweep.sh selects a device: inside the process the GPU is always
/// ordinal 0, but nvidia-smi still addresses it by its real index, and the CSV
/// must record which physical card produced the row.
int g_label = -1;

std::string nvsmi(const char* query) {
    char cmd[512];
    std::snprintf(cmd, sizeof(cmd),
                  "nvidia-smi --query-gpu=%s --format=csv,noheader,nounits -i %d 2>/dev/null",
                  query, g_label);
    FILE* p = popen(cmd, "r");
    if (!p) return "n/a";
    char buf[256] = {0};
    if (!fgets(buf, sizeof(buf), p)) { pclose(p); return "n/a"; }
    pclose(p);
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

/// GPU clock and thermal state.
///
/// Recorded per cell rather than once per run: a later cell running hotter
/// than an earlier one is a measurement artefact, and it is only visible if
/// the state is sampled alongside the timings. With a sweep this matters more
/// than it did with a single scene, because the run is far longer.
std::string gpu_state() { return nvsmi("clocks.sm,clocks.mem,temperature.gpu,power.draw"); }

/// Total device memory held by OTHER compute processes, in MiB.
///
/// Desktop compositing shows up here at a couple of hundred MiB and is
/// tolerable; a real workload is not. The threshold is deliberately generous:
/// the failure this guards against is catastrophic and silent, so a false
/// refusal costs far less than a false result.
long other_process_mib() {
    char cmd[256];
    std::snprintf(cmd, sizeof(cmd),
                  "nvidia-smi --query-compute-apps=pid,used_memory "
                  "--format=csv,noheader,nounits -i %d 2>/dev/null",
                  g_label);
    FILE* p = popen(cmd, "r");
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

uint32_t next_pow2_u(uint32_t v) {
    uint32_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

/// Device bytes one volume of this pool size costs.
///
/// Needed because the sweep's larger cells multiplied by BATCH volumes can
/// exceed an 8 GiB card, and silently falling back to a smaller BATCH without
/// saying so would change what is being measured.
size_t volume_bytes(int32_t pool_blocks) {
    const uint32_t table = next_pow2_u((uint32_t)pool_blocks * 2u) + 256u;
    return (size_t)table * 16u                                       // hash entries
           + (size_t)pool_blocks * kBlockVoxels * 5u * sizeof(float)  // tsdf, w, r, g, b
           + (size_t)pool_blocks * 3u * sizeof(int32_t)               // block_coord
           + 4096u;
}

struct Row {
    std::string cell, axis, arm, stage;
    int32_t points = 0, pool = 0, blocks = 0;
    double load_factor = 0;
    int batch = 0;
    Stats s;
    std::string gpu;
};

void emit(std::vector<Row>& rows, const Row& proto, const char* arm, const char* stage,
          const Stats& s) {
    Row r = proto;
    r.arm = arm;
    r.stage = stage;
    r.s = s;
    rows.push_back(r);
    std::printf("  %-14s %-9s  p50 %8.3f  p95 %8.3f  p99 %8.3f  min %8.3f  (n=%d)\n", arm, stage,
                s.p50, s.p95, s.p99, s.min, s.n);
}

// ---------------------------------------------------------------------------
// One cell
// ---------------------------------------------------------------------------

struct ArmHandles {
    bool have_a4 = false;
    bool have_a5 = false;
    OsnTritonKernel* k_upd = nullptr;
    OsnTritonKernel* k_alloc = nullptr;
};

/// Run every arm over one workload cell. Returns false if the cell was skipped
/// or failed a validity gate, which is reported rather than silently dropped.
bool run_cell(const Cell& cell, const ArmHandles& arms, int reps, int warmup, int batch_req,
              float voxel, std::vector<Row>& rows, bool want_extract) {
    const auto pts = make_points(cell);
    const int32_t n = (int32_t)(pts.size() / 3);

    // Pool sizing. Fixed cells state their own; load-factor cells measure the
    // block count first with a generous pool, then size from it. Measuring
    // beats predicting here: the block count depends on truncation band width
    // and on how the surface lands relative to the block grid, neither of
    // which is worth deriving by hand for every cell.
    int32_t pool = cell.pool_blocks;
    if (pool == 0) {
        OsnTsdfCudaVolume* probe = osn_tsdf_cuda_create(voxel, 1 << 18, -1.0f, 32.0f);
        if (!probe) {
            std::printf("  [%s] probe volume allocation failed\n", cell.name);
            return false;
        }
        float* dp = nullptr;
        cudaMalloc(&dp, pts.size() * sizeof(float));
        cudaMemcpy(dp, pts.data(), pts.size() * sizeof(float), cudaMemcpyHostToDevice);
        osn_tsdf_cuda_allocate_blocks(probe, dp, n, 0, 0, 0, 0);
        osn_tsdf_cuda_synchronize(probe);
        const int32_t nb = osn_tsdf_cuda_block_count(probe);
        cudaFree(dp);
        osn_tsdf_cuda_destroy(probe);
        if (nb <= 0) {
            std::printf("  [%s] probe found no blocks\n", cell.name);
            return false;
        }
        pool = (int32_t)std::max(64.0f, std::ceil(nb * cell.pool_factor));
    }

    // BATCH must fit. Reducing it is legitimate, since it only changes how many
    // launches share one completion, but it must be reported: a cell measured
    // at a different batch size is not directly comparable to one that was not.
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    const size_t per_vol = volume_bytes(pool);
    const size_t usable = (size_t)(free_b * 0.75);
    int batch = batch_req;
    while (batch > 1 && (size_t)batch * per_vol > usable) batch--;
    if ((size_t)batch * per_vol > usable) {
        std::printf("=== cell %s (axis %s) ===\n", cell.name, cell.axis);
        std::printf("  SKIPPED: one volume needs %.0f MiB, only %.0f MiB usable\n\n",
                    per_vol / 1048576.0, usable / 1048576.0);
        return false;
    }

    std::printf("=== cell %s (axis %s) ===\n", cell.name, cell.axis);
    std::printf("  scene %s extent %.2f m, %d points, voxel %.3f m\n", shape_name(cell.shape),
                cell.extent, n, voxel);
    std::printf("  pool %d blocks (%.0f MiB/volume), batch %d%s\n", pool, per_vol / 1048576.0,
                batch, batch != batch_req ? "  <-- REDUCED TO FIT" : "");

    float* d_pts = nullptr;
    cudaMalloc(&d_pts, pts.size() * sizeof(float));
    cudaMemcpy(d_pts, pts.data(), pts.size() * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<OsnTsdfCudaVolume*> vols((size_t)batch, nullptr);
    std::vector<OsnTsdfDeviceView> dvs((size_t)batch);
    for (int j = 0; j < batch; ++j) {
        vols[j] = osn_tsdf_cuda_create(voxel, pool, -1.0f, 32.0f);
        if (!vols[j]) {
            std::printf("  [%s] volume %d allocation failed\n\n", cell.name, j);
            for (auto* v : vols)
                if (v) osn_tsdf_cuda_destroy(v);
            cudaFree(d_pts);
            return false;
        }
        osn_tsdf_cuda_device_view(vols[j], &dvs[j]);
    }
    const uint32_t table_slots = dvs[0].hash_mask + 1u;

    auto reset_all = [&]() {
        for (int j = 0; j < batch; ++j) osn_tsdf_cuda_reset(vols[j]);
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
            osn_triton_launch(arms.k_alloc, grid, OSN_TRITON_ALLOC_BLOCK_DIM_X,
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
            osn_triton_launch(arms.k_upd, grid, OSN_TRITON_UPDATE_BLOCK_DIM_X,
                              OSN_TRITON_UPDATE_SHARED, args, 17);
        }
#endif
    };

    std::vector<float> mesh_buf((size_t)(4 << 20) * 6);
    std::vector<double> smp[3][3];
    int32_t blocks_seen[3] = {-1, -1, -1};
    uint64_t drops_seen[3] = {0, 0, 0};
    const char* arm_name[3] = {"A3-cuda", "A4-rust", "A5-triton"};
    GpuTimer t;

    for (int i = 0; i < warmup + reps; ++i) {
        for (int k = 0; k < 3; ++k) {
            const int arm = (i + k) % 3;
            if (arm == 1 && !arms.have_a4) continue;
            if (arm == 2 && !arms.have_a5) continue;

            // allocate: BATCH launches into BATCH empty volumes.
            reset_all();
            cudaDeviceSynchronize();
            t.start();
            for (int j = 0; j < batch; ++j) launch_alloc(arm, j);
            const double a = t.stop_ms() / batch;

            // update: the volumes are now allocated, so BATCH updates all take
            // the same path. Accumulating into an already-updated volume is
            // exactly what a multi-frame integrate does.
            t.start();
            for (int j = 0; j < batch; ++j) launch_update(arm, j);
            const double u = t.stop_ms() / batch;

            // Record what this arm actually built, every repetition rather than
            // once. Sampling it once would miss a cell where an arm drops
            // blocks only under contention, which is precisely the regime the
            // sweep exists to explore.
            osn_tsdf_cuda_synchronize(vols[0]);
            blocks_seen[arm] = osn_tsdf_cuda_block_count(vols[0]);
            drops_seen[arm] = osn_tsdf_cuda_drop_count(vols[0]);

            double e = -1;
            if (arm == 0 && want_extract) {
                // Extraction is shared across the GPU arms and does not mutate
                // the volume, so BATCH extracts of one volume are equivalent
                // and can share a window.
                t.start();
                for (int j = 0; j < batch; ++j)
                    osn_tsdf_cuda_extract_mesh(vols[0], mesh_buf.data(), nullptr, 4 << 20, 0.5f,
                                               0.0f);
                e = t.stop_ms() / batch;
            }

            if (i >= warmup) {
                smp[arm][0].push_back(a);
                smp[arm][1].push_back(u);
                if (e >= 0) smp[arm][2].push_back(e);
            }
        }
    }

    // ---- validity gates ---------------------------------------------------
    //
    // A cell that overflowed its pool, or where the arms disagree on what they
    // built, is not a measurement of anything. Report and discard rather than
    // publishing a fast number that came from doing less work: an arm that
    // drops blocks under contention would otherwise look like the winner.
    bool valid = true;
    for (int arm = 0; arm < 3; ++arm) {
        if (smp[arm][0].empty()) continue;
        if (drops_seen[arm] > 0) {
            // A drop is one lost point-voxel CONTRIBUTION, not one lost block:
            // many points probe toward the same block, so an unreachable block
            // produces many drops. At load factor 0.283 A5 reported 26,940
            // drops for 13 missing blocks. Two causes are possible and the
            // counter does not distinguish them: pool exhaustion, which every
            // arm reports, and probe exhaustion, which only A5 can suffer
            // because its probe bound is a compile-time constant.
            std::printf("  INVALID: %s dropped %llu contributions (pool %d full, or for A5 "
                        "the probe bound)\n",
                        arm_name[arm], (unsigned long long)drops_seen[arm], pool);
            valid = false;
        }
        if (blocks_seen[0] > 0 && blocks_seen[arm] > 0 && blocks_seen[arm] != blocks_seen[0]) {
            std::printf("  INVALID: %s built %d blocks, A3 built %d\n", arm_name[arm],
                        blocks_seen[arm], blocks_seen[0]);
            valid = false;
        }
    }

    const double lf = table_slots ? (double)blocks_seen[0] / (double)table_slots : 0.0;
    std::printf("  blocks %d / %u slots  ->  load factor %.3f%s\n", blocks_seen[0], table_slots, lf,
                valid ? "" : "   [CELL INVALID, NOT RECORDED]");

    if (valid) {
        Row proto;
        proto.cell = cell.name;
        proto.axis = cell.axis;
        proto.points = n;
        proto.pool = pool;
        proto.blocks = blocks_seen[0];
        proto.load_factor = lf;
        proto.batch = batch;
        proto.gpu = gpu_state();
        const char* stage_name[3] = {"allocate", "update", "extract"};
        for (int arm = 0; arm < 3; ++arm) {
            if (smp[arm][0].empty()) continue;
            for (int st = 0; st < 3; ++st)
                if (!smp[arm][st].empty())
                    emit(rows, proto, arm_name[arm], stage_name[st], summarize(smp[arm][st]));
        }

        std::printf("  drift (per-arm first vs second half p50, update):");
        for (int arm = 0; arm < 3; ++arm) {
            auto& v = smp[arm][1];
            if (v.size() < 4) continue;
            std::vector<double> h1(v.begin(), v.begin() + v.size() / 2);
            std::vector<double> h2(v.begin() + v.size() / 2, v.end());
            const double x = summarize(h1).p50, y = summarize(h2).p50;
            const double drift = x > 0 ? 100.0 * (y - x) / x : 0.0;
            std::printf("  %s %+.1f%%%s", arm_name[arm], drift,
                        std::fabs(drift) > 5.0 ? "(!)" : "");
        }
        std::printf("\n");
    }
    std::printf("\n");

    for (auto* v : vols)
        if (v) osn_tsdf_cuda_destroy(v);
    cudaFree(d_pts);
    return valid;
}

}  // namespace

int main(int argc, char** argv) {
    int reps = 30, warmup = 5, batch = 8;
    float voxel = 0.01f;
    std::string only_cells, only_axis, csv_path;
    bool want_cpu = false, want_extract = true;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--device") g_device = std::atoi(next());
        else if (a == "--device-label") g_label = std::atoi(next());
        else if (a == "--reps") reps = std::atoi(next());
        else if (a == "--warmup") warmup = std::atoi(next());
        else if (a == "--batch") batch = std::atoi(next());
        else if (a == "--voxel") voxel = (float)std::atof(next());
        else if (a == "--cells") only_cells = next();
        else if (a == "--axis") only_axis = next();
        else if (a == "--csv") csv_path = next();
        else if (a == "--cpu") want_cpu = true;
        else if (a == "--no-extract") want_extract = false;
        else if (a == "--list") {
            for (const auto& c : kDefaultCells) std::printf("%-12s %s\n", c.name, c.axis);
            return 0;
        } else {
            std::printf("usage: bench_arms [--device N] [--reps N] [--warmup N] [--batch N]\n"
                        "                  [--voxel M] [--cells a,b,c] [--axis NAME]\n"
                        "                  [--csv PATH] [--cpu] [--no-extract] [--list]\n");
            return 1;
        }
    }

    if (g_label < 0) g_label = g_device;

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
    // Must be set before any context exists, as must the device selection:
    // both the Triton cubin and the cuda-oxide module bind to the context
    // current at load time, so the device cannot change afterwards.
    cudaSetDevice(g_device);
    const cudaError_t flag_rc = cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    cudaFree(nullptr);  // force primary context creation on the chosen device

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, g_device);
    std::printf("=== TSDF fusion arm sweep ===\n");
    std::printf("  device %d (cuda ordinal %d): %s sm_%d%d | %d SMs | %.1f GiB\n", g_label,
                g_device, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.totalGlobalMem / 1073741824.0);
    std::printf("  reps %d (warmup %d discarded) | batch %d | voxel %.3f m\n", reps, warmup, batch,
                voxel);
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

    ArmHandles arms;
#ifdef OSN_TSDF_HAVE_A4
    arms.have_a4 = (a4_init() == 0);
#endif
#ifdef HAVE_A5
    const std::string dir =
        getenv("OSN_TRITON_DIR") ? getenv("OSN_TRITON_DIR") : "../artifacts/triton";
    arms.k_upd = osn_triton_load((dir + "/" + OSN_TRITON_UPDATE_NAME + ".cubin").c_str(),
                                 OSN_TRITON_UPDATE_NAME);
    arms.k_alloc = osn_triton_load((dir + "/" + OSN_TRITON_ALLOC_NAME + ".cubin").c_str(),
                                   OSN_TRITON_ALLOC_NAME);
    arms.have_a5 = (arms.k_upd && arms.k_alloc);
#endif
    std::printf("  arms: A3 yes | A4 %s | A5 %s\n", arms.have_a4 ? "yes" : "NO",
                arms.have_a5 ? "yes" : "NO");

    // Verify the arms did not move the context to another device.
    //
    // This guard exists because the failure it catches actually happened and
    // produced a full sweep of plausible, wrong numbers. Arm A4's loader calls
    // `CudaContext::new(0)` (crates/tsdf-rust-cuda/src/lib.rs), which hardcodes
    // device 0 and, being a driver-API context creation, becomes current for
    // the whole process. Every arm afterwards therefore ran on device 0 while
    // the harness printed device 1's name and SM count in the header. The
    // resulting table showed a 2.33x SM difference producing a 1.00x speedup,
    // which reads as a finding rather than as a bug.
    //
    // `cuCtxGetDevice` reports the context actually in force, unlike
    // `cudaGetDevice` which reports the runtime's intent. The robust fix is to
    // select the device with CUDA_VISIBLE_DEVICES so the process sees only one
    // GPU and a hardcoded ordinal 0 is correct by construction; tools/sweep.sh
    // does that. This check is the backstop.
    CUdevice ctx_dev = -1;
    if (cuCtxGetDevice(&ctx_dev) == CUDA_SUCCESS && (int)ctx_dev != g_device) {
        std::printf("\n  REFUSING TO RUN: asked for device %d, but the current CUDA context\n",
                    g_device);
        std::printf("  is on device %d. An arm's loader has hijacked the context.\n",
                    (int)ctx_dev);
        std::printf("  Select the device with CUDA_VISIBLE_DEVICES=%d and pass --device 0.\n",
                    g_device);
        return 5;
    }
    std::printf("  context device verified: %d\n\n", (int)ctx_dev);

    // Cell selection.
    std::vector<Cell> cells;
    for (const auto& c : kDefaultCells) {
        if (!only_axis.empty() && only_axis != c.axis) continue;
        if (!only_cells.empty()) {
            const std::string needle = std::string(",") + c.name + ",";
            if (("," + only_cells + ",").find(needle) == std::string::npos) continue;
        }
        cells.push_back(c);
    }
    if (cells.empty()) {
        std::printf("  no cells selected\n");
        return 1;
    }

    std::vector<Row> rows;
    int ok = 0, bad = 0;
    for (const auto& c : cells) {
        if (run_cell(c, arms, reps, warmup, batch, voxel, rows, want_extract)) ok++;
        else bad++;
    }

    // ---- A2: CPU C++ -------------------------------------------------------
    //
    // Opt-in and baseline-only. The CPU arm is orders of magnitude slower, so
    // sweeping it would dominate wall clock without adding information: it has
    // neither the hash-probe divergence story nor the atomics contention story
    // that the GPU axes exist to separate.
    if (want_cpu) {
        const auto pts = scenes::sphere(0.5f, 400, 800);
        const int32_t n = (int32_t)(pts.size() / 3);
        VolumeConfig cfg;
        cfg.voxel_size_m = voxel;
        cfg.pool_capacity_blocks = 1 << 14;
        std::vector<float> mesh_buf((size_t)(4 << 20) * 6);
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
            if (i > 0) {
                alloc.push_back(a);
                upd.push_back(u);
                ext.push_back(e);
            }
        }
        std::printf("=== cell base (axis cpu) ===\n--- A2 cpu (%d threads) ---\n",
                    Volume(cfg).thread_count());
        Row proto;
        proto.cell = "base";
        proto.axis = "cpu";
        proto.points = n;
        proto.pool = cfg.pool_capacity_blocks;
        proto.batch = 1;
        proto.gpu = gpu_state();
        emit(rows, proto, "A2-cpu", "allocate", summarize(alloc));
        emit(rows, proto, "A2-cpu", "update", summarize(upd));
        emit(rows, proto, "A2-cpu", "extract", summarize(ext));
        std::printf("\n");
    }

    std::printf("  cells: %d recorded, %d skipped or invalid\n", ok, bad);
    std::printf("  state at end:   %s\n", gpu_state().c_str());

    // Raw CSV, so the numbers can be re-analysed without re-running.
    // Appends when OSN_BENCH_CSV_APPEND is set, which is how the multi-device
    // driver accumulates one file across processes.
    if (csv_path.empty() && getenv("OSN_BENCH_CSV")) csv_path = getenv("OSN_BENCH_CSV");
    if (!csv_path.empty()) {
        const bool append = getenv("OSN_BENCH_CSV_APPEND") != nullptr;
        FILE* f = fopen(csv_path.c_str(), append ? "a" : "w");
        if (f) {
            if (!append || ftell(f) == 0)
                fprintf(f, "device,device_name,sm,sm_count,cell,axis,points,voxel_m,pool_blocks,"
                           "blocks,load_factor,batch,arm,stage,p50_ms,p95_ms,p99_ms,min_ms,"
                           "mean_ms,n,gpu_state\n");
            for (const auto& r : rows)
                fprintf(f,
                        "%d,\"%s\",%d%d,%d,%s,%s,%d,%.6f,%d,%d,%.6f,%d,%s,%s,"
                        "%.6f,%.6f,%.6f,%.6f,%.6f,%d,\"%s\"\n",
                        g_label, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                        r.cell.c_str(), r.axis.c_str(), r.points, voxel, r.pool, r.blocks,
                        r.load_factor, r.batch, r.arm.c_str(), r.stage.c_str(), r.s.p50, r.s.p95,
                        r.s.p99, r.s.min, r.s.mean, r.s.n, r.gpu.c_str());
            fclose(f);
            std::printf("  wrote %s (%zu rows%s)\n", csv_path.c_str(), rows.size(),
                        append ? ", appended" : "");
        }
    }

    if (arms.k_upd) osn_triton_unload(arms.k_upd);
    if (arms.k_alloc) osn_triton_unload(arms.k_alloc);
    return bad > 0 ? 4 : 0;
}
