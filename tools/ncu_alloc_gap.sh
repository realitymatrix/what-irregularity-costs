#!/usr/bin/env bash
# Profile the A3 and A4 allocate kernels and dump warp stall reasons.
#
# Ten mechanisms for the A4/A3 allocate gap have been tested and eliminated:
# register pressure, f32::clamp, read_volatile, static instruction count,
# dynamic compare-exchange count, memory-ordering scope, the shared counter,
# the fence, the publication store, and the publication spin. Every static and
# dynamic count favours A4, which still runs 1.4x to 1.7x slower.
#
# What has never been measured is where the warps actually wait. That is what
# this collects.
#
# REQUIRES GPU performance counters to be readable by non-root users. See
# docs/A3-A4-GAP.md for the one-time enable; without it ncu reports
# ERR_NVGPUCTRPERM and the run below produces nothing.
set -euo pipefail

BUILD="${1:-build}"
OUT="results/ncu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
NCU="${NCU:-/usr/local/cuda/bin/ncu}"

if ! "$NCU" --version >/dev/null 2>&1; then
    echo "error: $NCU not runnable" >&2
    exit 1
fi

# Fail loudly rather than producing an empty report: a profiler that silently
# collects nothing is exactly how a wrong conclusion gets published.
probe="$BUILD/probe_cas128_check"
if ! CUDA_VISIBLE_DEVICES=0 "$NCU" --section WarpStateStats -k "regex:alloc_kernel" \
        --launch-count 1 "$probe" 2>&1 | grep -q "ERR_NVGPUCTRPERM"; then
    echo "counters readable, collecting"
else
    cat >&2 <<'MSG'
error: GPU performance counters are not readable by this user (ERR_NVGPUCTRPERM).

Enable them once, as root, then reboot:

    echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
      | sudo tee /etc/modprobe.d/nvidia-profiling.conf
    sudo update-initramfs -u
    sudo reboot

Or, without a reboot, run this script under sudo:

    sudo -E NCU=/usr/local/cuda/bin/ncu tools/ncu_alloc_gap.sh build

Note the modprobe change lets any local user read GPU counters, which NVIDIA
restricts by default for a reason. On a shared machine prefer the sudo form.
MSG
    exit 2
fi

# Both arms in one process so clocks, allocation and input are identical.
# WarpStateStats is the section that answers "where does the warp wait";
# SpeedOfLight and MemoryWorkloadAnalysis bound whether it is issue, memory or
# latency limited.
CUDA_VISIBLE_DEVICES=0 "$NCU" \
    --target-processes all \
    -k "regex:alloc_kernel" \
    --section WarpStateStats \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section Occupancy \
    --csv --log-file "$OUT/ncu.csv" \
    "$probe" > "$OUT/stdout.txt" 2>&1 || true

CUDA_VISIBLE_DEVICES=0 "$NCU" \
    --target-processes all \
    -k "regex:alloc_kernel" \
    --section WarpStateStats \
    --section SpeedOfLight \
    "$probe" > "$OUT/report.txt" 2>&1 || true

echo "wrote $OUT/report.txt and $OUT/ncu.csv"
echo
echo "The line that matters is the top stall reason per kernel. If A4's warps"
echo "wait on a different reason than A3's, that is the mechanism. If they wait"
echo "on the SAME reason for longer, it is throughput, not structure."
grep -E "alloc_kernel|Stall|Warp Cycles|Achieved Occupancy" "$OUT/report.txt" | head -40 || true
