"""Ahead-of-time export of Triton kernels to cubin + launch manifest.

Why: the Triton fusion arm must be launched from the same Rust driver as every
other arm. If Triton kernels were launched from embedded Python, the measured
path would include an interpreter round-trip and the comparison would be
measuring language runtime rather than kernel time. Triton exposes the compiled
cubin, so Python can be a *build-time* dependency only.

This script compiles the kernels and writes, per kernel:
    <name>.cubin   the compiled module
    <name>.json    everything the Rust launcher needs

The launch ABI is not documented; it was read out of
triton/backends/nvidia/driver.py in Triton 3.6.0:

  * Kernel params are the non-constexpr arguments in declaration order,
    followed by `&global_scratch` then `&profile_scratch` (driver.py:261-262).
    Both are CUdeviceptr and may be null when their sizes are zero.
  * blockDimX = 32 * num_warps, blockDimY = blockDimZ = 1 (driver.py:330-332).
    NOTE: this is NOT the BLOCK constexpr. BLOCK is a tile size; the CUDA block
    is derived from num_warps. Confusing the two launches the wrong shape and
    silently produces wrong results rather than an error.
  * `shared` from metadata is the dynamic shared-memory size.

Re-verify these against driver.py on any Triton upgrade. A change here fails
silently, so tests/validate_abi must be run after any bump.

Usage:
    python tools/triton_aot.py <outdir>
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import torch
import triton

REPO = pathlib.Path(__file__).resolve().parent.parent
SPIKE = REPO / "spikes" / "s2_triton" / "s2_triton_spike.py"


def load_kernels():
    spec = importlib.util.spec_from_file_location("s2", SPIKE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def manifest_for(compiled, constexprs: dict) -> dict:
    md = compiled.metadata
    sig = compiled.src.signature

    # Runtime (non-constexpr) parameters, in declaration order. This order is
    # the ABI: the Rust launcher builds its param array to match.
    runtime_args = [
        {"name": n, "type": t} for n, t in sig.items() if t != "constexpr"
    ]

    return {
        "name": md.name,
        "arch": str(md.arch),
        "num_warps": md.num_warps,
        "num_stages": md.num_stages,
        "block_dim_x": 32 * md.num_warps,  # driver.py:330
        "shared_bytes": md.shared,
        "global_scratch_size": md.global_scratch_size,
        "global_scratch_align": md.global_scratch_align,
        "profile_scratch_size": getattr(md, "profile_scratch_size", 0),
        "runtime_args": runtime_args,
        # Trailing params appended by the launcher after the runtime args.
        "trailing_params": ["global_scratch", "profile_scratch"],
        "constexprs": constexprs,
        "triton_version": triton.__version__,
    }


def main(argv: list[str]) -> int:
    outdir = pathlib.Path(argv[1] if len(argv) > 1 else REPO / "artifacts" / "triton")
    outdir.mkdir(parents=True, exist_ok=True)

    m = load_kernels()
    dev = "cuda"
    grid = (triton.cdiv(m.THREADS, m.BLOCK),)

    # Compile by running once on real buffers. Deliberately not torch.randn or
    # zeros-as-placeholder: these are the same shapes and sentinels the kernels
    # see in production, so the compiled specialisation matches.
    table = torch.full((m.HASH_SIZE + 1,), m.HASH_EMPTY, dtype=torch.int32, device=dev)
    table[m.HASH_SIZE] = m.SCRATCH_SENTINEL
    slot = torch.zeros(m.THREADS, dtype=torch.int32, device=dev)
    won = torch.zeros(m.THREADS, dtype=torch.int32, device=dev)

    hash_cx = {
        "HASH_SIZE": m.HASH_SIZE,
        "HASH_EMPTY": m.HASH_EMPTY,
        "DUP": m.DUP,
        "MAX_PROBE": m.MAX_PROBE,
        "BLOCK": m.BLOCK,
    }
    k_hash = m.hash_insert_kernel[grid](table, slot, won, m.THREADS, **hash_cx)

    sum_xyz = torch.zeros(m.HASH_SIZE * 3, dtype=torch.float32, device=dev)
    weight = torch.zeros(m.HASH_SIZE, dtype=torch.float32, device=dev)
    count = torch.zeros(m.HASH_SIZE, dtype=torch.int32, device=dev)

    acc_cx = {"HASH_EMPTY": m.HASH_EMPTY, "BLOCK": m.BLOCK}
    k_acc = m.voxel_accumulate_kernel[grid](
        slot, sum_xyz, weight, count, m.THREADS, **acc_cx
    )

    written = []
    for compiled, cx in ((k_hash, hash_cx), (k_acc, acc_cx)):
        man = manifest_for(compiled, cx)
        name = man["name"]
        (outdir / f"{name}.cubin").write_bytes(compiled.asm["cubin"])
        (outdir / f"{name}.json").write_text(json.dumps(man, indent=2) + "\n")
        written.append(name)
        print(f"{name}:")
        print(f"  cubin        {len(compiled.asm['cubin'])} bytes")
        print(f"  block_dim_x  {man['block_dim_x']}  (num_warps {man['num_warps']})")
        print(f"  shared       {man['shared_bytes']} bytes")
        print(f"  runtime args {[a['name'] for a in man['runtime_args']]}")
        print(f"  + trailing   {man['trailing_params']}")

    # Reference values for the Rust ABI validator. If a Rust launch reproduces
    # these exactly, the argument packing, block shape and shared-memory size
    # are all correct. Anything else is a silent ABI bug.
    torch.cuda.synchronize()
    expected = {
        "grid_x": grid[0],
        "n_threads": m.THREADS,
        "hash_size": m.HASH_SIZE,
        "hash_empty": m.HASH_EMPTY,
        "scratch_sentinel": m.SCRATCH_SENTINEL,
        "n_coords": m.N_COORDS,
        "dup": m.DUP,
        "winners": int(won.sum()),
        "unresolved": int((slot == m.HASH_EMPTY).sum()),
        "distinct_slots": int(torch.unique(slot.view(m.N_COORDS, m.DUP)[:, 0]).numel()),
        "total_count": int(count.sum()),
        "total_weight": float(weight.sum()),
        "total_sum_x": float(sum_xyz[0::3].sum()),
        "total_sum_z": float(sum_xyz[2::3].sum()),
    }
    (outdir / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    print(f"\nwrote {len(written)} kernels + expected.json to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
