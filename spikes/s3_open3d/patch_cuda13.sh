#!/usr/bin/env bash
# Make stdgpu (pinned by Open3D 0.19 to a 2023 commit) build against CUDA 13.
#
# CUDA 13 removed the long-deprecated cudaDeviceProp::clockRate field. stdgpu
# reads it only to pretty-print device info in print_device_information(), so
# the value is cosmetic. Query it through cudaDeviceGetAttribute instead, which
# is still supported.
#
# Idempotent: exits successfully if the patch is already applied, so repeated
# ExternalProject patch steps are harmless.
set -euo pipefail

src="$1"
f="${src}/src/stdgpu/cuda/impl/device.cpp"

if [[ ! -f "$f" ]]; then
    echo "patch_cuda13: ${f} not found" >&2
    exit 1
fi

if grep -q 'cudaDevAttrClockRate' "$f"; then
    echo "patch_cuda13: already applied"
    exit 0
fi

if ! grep -q 'properties\.clockRate' "$f"; then
    echo "patch_cuda13: nothing to patch (clockRate absent); assuming newer stdgpu"
    exit 0
fi

python3 - "$f" <<'PY'
import sys

path = sys.argv[1]
old = "static_cast<float>(properties.clockRate)"
new = ("static_cast<float>([]{ int khz = 0; int dev = 0; "
       "cudaGetDevice(&dev); "
       "cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev); "
       "return khz; }())")

text = open(path).read()
if old not in text:
    raise SystemExit(f"patch_cuda13: pattern not found in {path}")
open(path, "w").write(text.replace(old, new))
print("patch_cuda13: applied clockRate -> cudaDeviceGetAttribute")
PY
