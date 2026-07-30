# Bug in the reference TSDF: device integrate ignores mesh mode

Found 2026-07-30 while wiring arm A0. Affects
`openstrate-reconstruct-rs/vendor/libinfer/src/tsdf_hash.cu` upstream, not just
the vendored copy here.

## Symptom

With mesh mode enabled, integrating through the device-pointer entry point and
then extracting a mesh returns **zero vertices**, while every other diagnostic
looks healthy:

```
points integrated  2938758
blocks allocated   1203
points dropped     0
mesh               0 verts, 0 tris     <-- silent
```

## Cause

`tsdf_hash_add_points_chunk` (host pointers) branches on `mesh_mode`:

```c
if (h->mesh_mode) {
    hash_integrate_tsdf_kernel<<<...>>>(...);   // projective TSDF
    return 0;
}
hash_integrate_kernel<<<...>>>(...);            // centroid binning
```

`tsdf_hash_add_points_chunk_device` does **not**. It unconditionally launches
`hash_integrate_kernel`, so no signed distance field is ever built and marching
cubes finds no zero crossing.

`tsdf_hash_set_mesh_mode` returns 0 (success), the integrate call returns 0, the
block pool fills normally, and the drop count stays at zero. Nothing indicates
the mode was ignored.

## Impact beyond this project

The device path exists to avoid a host round-trip, and the header recommends it
for the chunk loop:

> Designed for the chunk-loop case where backproject already produces
> world-space points on device [...] eliminates ~50-80 ms/chunk of marshalling
> + PCIe transfer.

So any OSN pipeline that combines mesh mode with the device fast path gets an
empty mesh with no error. Worth grepping the reconstruction stack for that
combination.

## Fix

Give the device variant the same `mesh_mode` branch as the host variant. The
kernel it needs, `hash_integrate_tsdf_kernel`, already accepts device pointers,
so the change is the branch, not new kernel work.

## Workaround in this repo

`ReferenceBackend::use_host_path` stages device -> host -> host-entry-point so
A0 builds a real signed field. Wasteful and deliberately temporary: it exists so
A0 can serve as the correctness oracle now, and should be removed once the
upstream branch is fixed and the vendored copy re-synced.

**Do not fix this by editing the vendored copy alone.** A0 is the oracle
precisely because it is upstream's code verbatim; a local-only fix would make
the oracle diverge from the implementation it is supposed to certify.
