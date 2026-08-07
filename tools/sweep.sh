#!/usr/bin/env bash
# Run the workload matrix across every visible GPU, into one CSV.
#
# One process per device, because both the Triton cubin and the cuda-oxide
# module bind to the CUDA context current at load time. Switching devices
# inside a process would need a reload per arm per device; a process per device
# removes the failure mode.
#
# Devices are run SEQUENTIALLY and never concurrently. The harness refuses to
# measure a shared device, and two sweeps at once would contend for host CPU
# (the driver is set to spin-wait) even though the GPUs are distinct. A prior
# run measured a 100x error from device sharing, so the cost of being careful
# here is trivially justified.
#
# Usage:
#   tools/sweep.sh [build-dir] [extra args passed to bench_arms...]
#
# Examples:
#   tools/sweep.sh build
#   tools/sweep.sh build --axis loadfactor --reps 50
set -euo pipefail

BUILD="${1:-build}"
shift || true

BIN="$BUILD/bench_arms"
[[ -x "$BIN" ]] || { echo "no $BIN; configure and build first" >&2; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="results/sweep_${STAMP}"
mkdir -p "$OUT"
CSV="$OUT/sweep.csv"

# Absolute, because the harness is run from the build directory so its default
# --triton-dir relative path resolves.
CSV_ABS="$(cd "$(dirname "$CSV")" && pwd)/$(basename "$CSV")"
BIN_ABS="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"

# Passes over the whole matrix. More than one is not optional for a published
# number: within a single pass the allocate stage looks tighter than it is, and
# cell r-2.0 reported a p50 of 0.047 ms in one pass against 0.060 in three
# reruns without any within-pass statistic flagging it.
REPEATS="${SWEEP_REPEATS:-3}"

mapfile -t DEVICES < <(nvidia-smi --query-gpu=index --format=csv,noheader)
echo "sweeping ${#DEVICES[@]} device(s) x $REPEATS pass(es) -> $CSV"

# Record the machine state alongside the numbers. Without this the CSV is not
# reproducible six months from now: driver and toolkit versions change what the
# compilers emit, which is the entire subject of the comparison.
{
    echo "date:    $(date -Is)"
    echo "host:    $(hostname)"
    echo "kernel:  $(uname -r)"
    echo "driver:  $(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0)"
    echo "nvcc:    $(nvcc --version 2>/dev/null | tail -1)"
    echo "rustc:   $(rustc --version 2>/dev/null || echo n/a)"
    echo "commit:  $(git rev-parse HEAD 2>/dev/null || echo n/a)"
    echo "dirty:   $(git status --porcelain 2>/dev/null | wc -l) file(s)"
    echo "args:    $*"
    echo
    nvidia-smi --query-gpu=index,name,compute_cap,memory.total,clocks.max.sm \
               --format=csv
} > "$OUT/environment.txt"

first=1
for run in $(seq 1 "$REPEATS"); do
for d in "${DEVICES[@]}"; do
    echo
    echo "################ device $d, pass $run ################"
    LOG="$OUT/device_${d}_run${run}.log"
    # First invocation writes the header, the rest append.
    if [[ "$first" == "1" ]]; then unset OSN_BENCH_CSV_APPEND; first=0
    else export OSN_BENCH_CSV_APPEND=1; fi
    # Select the device with CUDA_VISIBLE_DEVICES, not --device.
    #
    # Arm A4's loader calls CudaContext::new(0), which hardcodes device 0 and
    # makes that context current for the whole process, so --device alone is
    # silently ignored by every arm: an earlier sweep reported device 1's name
    # and SM count in the header while running everything on device 0, and the
    # resulting 1.00x "device scaling" looked like a result. Masking the process
    # down to one visible GPU makes ordinal 0 correct by construction, whatever
    # any dependency hardcodes. bench_arms verifies with cuCtxGetDevice and
    # refuses to run on a mismatch.
    #
    # `|| true` on the pipeline, because `set -e` with `pipefail` would abort
    # the whole sweep before PIPESTATUS is ever read. rc 4 means some cells were
    # invalid, which is an expected outcome worth continuing past, not a fault.
    ( cd "$BUILD" && CUDA_VISIBLE_DEVICES="$d" "$BIN_ABS" --device 0 --device-label "$d" \
        --run-id "$run" --csv "$CSV_ABS" "$@" ) 2>&1 | tee "$LOG" || true
    rc="${PIPESTATUS[0]}"
    # rc 4 means some cells were invalid or skipped, which is reported in the
    # log and is not a reason to abandon the remaining devices.
    [[ "$rc" == "0" || "$rc" == "4" ]] || { echo "device $d failed (rc=$rc)" >&2; exit "$rc"; }
done
done

echo
echo "wrote $CSV"
echo "analyse: tools/sweep_report.py $CSV"
