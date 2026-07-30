# Vendored reference TSDF (arm A0)

Copied verbatim from `openstrate-reconstruct-rs/vendor/libinfer/src/` on
2026-07-30. Petr's own code, so no third-party licence applies.

Verbatim on purpose. A0 is the correctness oracle every other arm is gated
against, so it must be the *same* implementation, not a port of it. Any edit
here weakens the oracle. If these need to change, change them upstream and
re-copy.

The headers are already a pure C ABI (opaque handle, scalars, raw pointers)
with no Rust types, no ARCore state and no TensorRT, which is why A0 needs only
a binding rather than a reimplementation.

  tsdf.h / tsdf.cu            dense sliding-window TSDF, 3 kernels. Context only;
                              the ported arms implement the hash volume.
  tsdf_hash.h / tsdf_hash.cu  sparse voxel-block-hashed TSDF, 10 kernels. This is
                              arm A0 and the algorithm all arms implement.
