"""Load a TensorRT engine, run it, and time it.

Confirms the engine is usable on this machine before any runner is written
against it, and gives the depth-stage latency the crossover claim needs.

Reports p50/p95/p99 with warmup discarded, matching the fusion harness, so the
depth and fusion numbers can be put in the same table.

Usage: python tools/verify_engine.py <engine> [reps]
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import tensorrt as trt
import torch

# Device buffers come from torch rather than the cuda bindings: torch is
# already a dependency, and `cuda.cudart` was renamed to
# `cuda.bindings.runtime` in newer cuda-python, so the import is a moving
# target. This is a verification tool, not the measured path; the real runner
# owns its own memory.


def main(argv: list[str]) -> int:
    path = pathlib.Path(argv[1])
    reps = int(argv[2]) if len(argv) > 2 else 50
    warmup = 10

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        print("FAILED to deserialize; engine architecture likely does not match this GPU")
        return 1
    ctx = engine.create_execution_context()
    print(f"loaded {path.name} ({path.stat().st_size / 1e6:.0f} MB)")

    torch_dtype = {np.float32: torch.float32, np.float16: torch.float16,
                   np.int32: torch.int32, np.int64: torch.int64}
    bufs = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        np_dt = trt.nptype(engine.get_tensor_dtype(name))
        td = torch_dtype[np_dt]
        mode = engine.get_tensor_mode(name)
        # Real-valued input rather than zeros: TensorRT does not branch on
        # values, but all-zero inputs can hit denormal paths in some layers.
        t = (torch.rand(shape, dtype=td, device="cuda") * 0.5 if mode == trt.TensorIOMode.INPUT
             else torch.zeros(shape, dtype=td, device="cuda"))
        bufs[name] = t
        ctx.set_tensor_address(name, t.data_ptr())
        print(f"  {'in ' if mode == trt.TensorIOMode.INPUT else 'out'} {name:14s} {shape} "
              f"{np.dtype(np_dt).name}  {t.numel() * t.element_size()/1e6:.1f} MB")

    stream = torch.cuda.Stream()

    times = []
    for i in range(warmup + reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ok = ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        if not ok:
            print("  execute failed")
            return 2
        if i >= warmup:
            times.append(dt)

    times.sort()
    def pct(p): return times[min(len(times) - 1, int(len(times) * p))]
    print(f"\n  inference, n={len(times)} (warmup {warmup} discarded)")
    print(f"    p50 {pct(0.5):7.3f} ms   p95 {pct(0.95):7.3f} ms   "
          f"p99 {pct(0.99):7.3f} ms   min {times[0]:7.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
