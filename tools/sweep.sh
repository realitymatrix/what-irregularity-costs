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

mapfile -t DEVICES < <(nvidia-smi --query-gpu=index --format=csv,noheader)
echo "sweeping ${#DEVICES[@]} device(s) -> $CSV"

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

for d in "${DEVICES[@]}"; do
    echo
    echo "################ device $d ################"
    LOG="$OUT/device_${d}.log"
    # First device writes the header, the rest append.
    if [[ "$d" == "${DEVICES[0]}" ]]; then unset OSN_BENCH_CSV_APPEND
    else export OSN_BENCH_CSV_APPEND=1; fi
    ( cd "$BUILD" && "$BIN_ABS" --device "$d" --csv "$CSV_ABS" "$@" ) 2>&1 | tee "$LOG"
    rc="${PIPESTATUS[0]}"
    # rc 4 means some cells were invalid or skipped, which is reported in the
    # log and is not a reason to abandon the remaining devices.
    [[ "$rc" == "0" || "$rc" == "4" ]] || { echo "device $d failed (rc=$rc)" >&2; exit "$rc"; }
done

echo
echo "wrote $CSV"
echo "analyse: tools/sweep_report.py $CSV"
