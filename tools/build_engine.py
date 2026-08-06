"""Build a TensorRT engine from ONNX for the local architecture.

Uses the TensorRT Python API rather than `trtexec`, because TensorRT here is a
pip wheel: `tensorrt_libs` ships the shared objects (including the sm_120
builder resource) but neither the `trtexec` binary nor the headers.

Engines are architecture-specific and are NOT committed. Build on the machine
that will run them: loading an engine built for another architecture either
fails or falls back to PTX JIT, and a JIT-vs-native mismatch is exactly the
confound the fusion arms were careful to avoid.

Usage:
    python tools/build_engine.py <model.onnx> <out.engine> [--fp32]
"""

from __future__ import annotations

import pathlib
import sys
import time

import tensorrt as trt


def build(onnx_path: pathlib.Path, out_path: pathlib.Path, fp16: bool = True) -> int:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    print(f"parsing {onnx_path.name} ({onnx_path.stat().st_size / 1e6:.0f} MB)")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  parse error: {parser.get_error(i)}")
            return 1

    print(f"  network: {network.num_layers} layers, {network.num_inputs} inputs, "
          f"{network.num_outputs} outputs")
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"    in  {t.name:16s} {t.shape} {t.dtype}")
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print(f"    out {t.name:16s} {t.shape} {t.dtype}")

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if fp16:
        if not builder.platform_has_fast_fp16:
            print("  WARNING: platform reports no fast fp16")
        cfg.set_flag(trt.BuilderFlag.FP16)
    # Highest optimisation level: this is a build-time cost paid once, and the
    # engine is what gets measured.
    try:
        cfg.builder_optimization_level = 3
    except AttributeError:
        pass

    # Pin every dynamic dimension. A dynamic engine would let TensorRT pick a
    # different kernel per shape, so two runs at the same resolution could use
    # different code and the timings would not be comparable.
    profile = builder.create_optimization_profile()
    dynamic = False
    for i in range(network.num_inputs):
        t = network.get_input(i)
        shape = [1 if d < 0 else d for d in t.shape]
        if any(d < 0 for d in t.shape):
            dynamic = True
        profile.set_shape(t.name, shape, shape, shape)
    if dynamic:
        print(f"  dynamic dims pinned to batch 1")
    cfg.add_optimization_profile(profile)

    print("  building (this takes minutes; TensorRT autotunes every layer)")
    t0 = time.time()
    plan = builder.build_serialized_network(network, cfg)
    if plan is None:
        print("  BUILD FAILED")
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # build_serialized_network returns IHostMemory, which is buffer-like but
    # has no __len__; memoryview gives both the bytes and the size.
    blob = memoryview(plan)
    out_path.write_bytes(blob)
    print(f"  wrote {out_path} ({blob.nbytes / 1e6:.0f} MB) in {time.time() - t0:.0f}s")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    return build(pathlib.Path(argv[1]), pathlib.Path(argv[2]), fp16="--fp32" not in argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
