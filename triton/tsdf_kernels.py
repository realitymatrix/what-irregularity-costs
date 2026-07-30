"""Triton kernels for the TSDF integrate path (arms A5a and A5b).

Same algorithm as the CUDA C++ (A3) and Rust (A4) arms, over the same device
memory: same packed 64-bit hash key, same open-addressed table with linear
probing, same truncation-band walk, same voxel-centre signed distance, same
weighted-sum accumulation. `tl.atomic_cas` on int64 emits
`atom.global.acq_rel.gpu.cas.b64`, which is the instruction A3 and A4 emit too,
so the comparison measures how each language expresses the algorithm rather
than which primitive it managed to reach.

Two arms share these kernels:

  A5a  triton-shared  update only; allocation runs on A3's CUDA kernel.
  A5b  triton-full    allocation and update both here.

`A5b - A5a` is then a direct measurement of Triton's control-flow tax on the
real workload, rather than an inference from the synthetic contention test in
spike S2.

## The tax, and why it is structural

Triton has no per-lane early exit. A probe loop must run to a fixed bound with
a `done` mask, so every lane pays worst-case probe cost even when it resolves
on the first try. `tl.atomic_cas` also takes no `mask` (unlike `tl.atomic_add`),
so resolved lanes cannot be masked off and must instead be aimed at a scratch
slot holding a sentinel that can never satisfy the compare. Both are worked
around here rather than avoided, because avoiding them would mean not
implementing the algorithm.

## Device memory layout

Fixed by the C++ library (see include/osn_tsdf/c_api.h). One hash slot is 16
bytes: an int64 key then an int32 block index then 4 bytes of padding. The same
allocation is therefore addressed two ways:

    table_i64[slot * 2]      the key
    table_i32[slot * 4 + 2]  the block index

Kernels take both pointers to the same buffer. Triton has no union type, and
bit-casting inside the kernel would cost more than passing a second pointer.
"""

import triton
import triton.language as tl

# Must match the packing in include/osn_tsdf/types.hpp.
#
# Instantiated as `tl.constexpr(...)`, not annotated `: tl.constexpr`. Triton
# rejects the annotation form for module-level globals accessed from a @jit
# function, and the error surfaces at the use site rather than the definition.
COORD_BIAS = tl.constexpr(1 << 20)
COORD_BITS = tl.constexpr(21)
EMPTY_KEY = tl.constexpr(-1)


@triton.jit
def _pack(x, y, z):
    """Pack a block coordinate into one 64-bit key.

    The whole coordinate must live in one word so a single compare-exchange
    publishes all of it. Splitting it lets a reader match part of a slot,
    mismatch on the rest, and probe onward; at GPU thread counts that becomes a
    full-table scan. The C++ arms hit exactly that before the layout changed.
    """
    xb = x.to(tl.int64) + COORD_BIAS
    yb = y.to(tl.int64) + COORD_BIAS
    zb = z.to(tl.int64) + COORD_BIAS
    return (xb << (2 * COORD_BITS)) | (yb << COORD_BITS) | zb


@triton.jit
def _hash(x, y, z, mask):
    """Same multiply-xor mix as the other arms. Must stay bit-identical: a
    different mix changes probe sequences, which changes which blocks share
    cache lines, and that would read as a language difference."""
    h = (x * 73856093) ^ (y * 19349663) ^ (z * 83492791)
    return h & mask


@triton.jit
def _floor_div(a, b):
    """floor(a / b), for POSITIVE b only.

    Truncating division mirrors block coordinates across the origin and puts a
    seam at exactly x = 0, so the correction matters for any scan that crosses
    it. `b` is the block dimension and always positive, so the sign test is on
    `a` alone: writing `(a < 0) != (b < 0)` would compare a tensor against a
    Python bool, since `b` arrives as a constexpr, and Triton rejects that."""
    q = a // b
    r = a % b
    return tl.where((r != 0) & (a < 0), q - 1, q)


@triton.jit
def tsdf_alloc_kernel(
    positions_ptr,
    table_i64,
    table_i32,
    block_count_ptr,
    block_coord_ptr,
    drop_count_ptr,
    scratch_slot,          # index of the sentinel slot, see module docstring
    n_points,
    hash_mask,
    pool_capacity,
    voxel_size,
    trunc,
    cam_x, cam_y, cam_z,
    radius_sq,
    BLOCK: tl.constexpr,
    STEPS: tl.constexpr,
    MAX_PROBE: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """Pass 1 (arm A5b only): claim hash slots for the blocks this batch touches."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    live = offs < n_points

    px = tl.load(positions_ptr + offs * 3 + 0, mask=live, other=0.0)
    py = tl.load(positions_ptr + offs * 3 + 1, mask=live, other=0.0)
    pz = tl.load(positions_ptr + offs * 3 + 2, mask=live, other=0.0)

    dx = px - cam_x
    dy = py - cam_y
    dz = pz - cam_z
    d2 = dx * dx + dy * dy + dz * dz
    dist = tl.sqrt(d2)
    live = live & (dist > 1e-6)
    if radius_sq > 0.0:
        live = live & (d2 <= radius_sq)
    inv_dist = tl.where(dist > 1e-6, 1.0 / dist, 0.0)
    ux = dx * inv_dist
    uy = dy * inv_dist
    uz = dz * inv_dist

    inv_voxel = 1.0 / voxel_size
    for s in tl.static_range(-STEPS, STEPS + 1):
        t = s * voxel_size
        vx = tl.floor((px + ux * t) * inv_voxel).to(tl.int32)
        vy = tl.floor((py + uy * t) * inv_voxel).to(tl.int32)
        vz = tl.floor((pz + uz * t) * inv_voxel).to(tl.int32)

        # Signed distance at the VOXEL CENTRE, not at the ray sample that
        # selected it. Points are binned with floor(p / voxel), so the centre is
        # up to half a voxel away; using the sample's own offset biases every
        # voxel and shows up as a uniform radius error on curved surfaces.
        cx = (vx.to(tl.float32) + 0.5) * voxel_size
        cy = (vy.to(tl.float32) + 0.5) * voxel_size
        cz = (vz.to(tl.float32) + 0.5) * voxel_size
        ex = cx - cam_x
        ey = cy - cam_y
        ez = cz - cam_z
        sdf = dist - tl.sqrt(ex * ex + ey * ey + ez * ez)

        # Same occlusion cull as the update pass. Applying it in only one pass
        # leaves blocks that never receive a contribution: meshes still match,
        # but block counts diverge from the other arms.
        active = live & (sdf >= -trunc)

        bx = _floor_div(vx, BLOCK_DIM)
        by = _floor_div(vy, BLOCK_DIM)
        bz = _floor_div(vz, BLOCK_DIM)
        want = _pack(bx, by, bz)
        start = _hash(bx, by, bz, hash_mask)

        done = ~active
        for p in tl.static_range(0, MAX_PROBE):
            slot = (start + p) & hash_mask
            # Resolved lanes are aimed at the scratch slot, whose key can never
            # equal `want` and can never be EMPTY. tl.atomic_cas takes no mask,
            # so this is the only way to make a lane inert.
            eff = tl.where(done, scratch_slot, slot)
            key = tl.load(table_i64 + eff * 2)

            hit = (key == want) & (~done)
            done = done | hit

            empty = (key == EMPTY_KEY) & (~done)
            eff_cas = tl.where(empty, slot, scratch_slot)
            cmp = tl.full((BLOCK,), EMPTY_KEY, tl.int64)
            old = tl.atomic_cas(table_i64 + eff_cas * 2, cmp, want)

            won = empty & (old == EMPTY_KEY)
            # Claim a pool index for the winners only.
            #
            # The counter is a single scalar, but the atomic must be issued as a
            # TENSOR of pointers. Passing the bare scalar pointer with a tensor
            # value crashes the compiler outright (MLIR assertion
            # "only integers and floats have a bitwidth"), rather than reporting
            # a type error, so the broadcast here is load-bearing.
            lane0 = tl.zeros((BLOCK,), tl.int32)
            idx = tl.atomic_add(block_count_ptr + lane0, 1, mask=won)
            ok = won & (idx < pool_capacity)
            # Publish coordinates then the index. The other arms fence between
            # the two; Triton's atomics are already release-ordered, and the
            # index store is the last write, so a reader that sees the index
            # sees the coordinates.
            tl.store(block_coord_ptr + idx * 3 + 0, bx, mask=ok)
            tl.store(block_coord_ptr + idx * 3 + 1, by, mask=ok)
            tl.store(block_coord_ptr + idx * 3 + 2, bz, mask=ok)
            tl.store(table_i32 + slot * 4 + 2, idx, mask=ok)

            # Pool exhausted: release the slot and report rather than dropping
            # silently. A saturating pool otherwise presents as a quality
            # regression when it is a capacity problem.
            over = won & (idx >= pool_capacity)
            tl.atomic_add(drop_count_ptr + lane0.to(tl.int64), 1, mask=over)
            empty_v = tl.full((BLOCK,), EMPTY_KEY, tl.int64)
            tl.store(table_i64 + slot * 2, empty_v, mask=over)

            done = done | won
            # Lost the race but the winner wrote our key: share the slot.
            done = done | (empty & (old == want))


@triton.jit
def tsdf_update_kernel(
    positions_ptr,
    table_i64,
    table_i32,
    tsdf_ptr,
    weight_ptr,
    n_points,
    hash_mask,
    voxel_size,
    trunc,
    weight_cap,
    cam_x, cam_y, cam_z,
    radius_sq,
    BLOCK: tl.constexpr,
    STEPS: tl.constexpr,
    MAX_PROBE: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    BLOCK_VOXELS: tl.constexpr,
):
    """Pass 2 (arms A5a and A5b): accumulate weighted sums into allocated blocks."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    live = offs < n_points

    px = tl.load(positions_ptr + offs * 3 + 0, mask=live, other=0.0)
    py = tl.load(positions_ptr + offs * 3 + 1, mask=live, other=0.0)
    pz = tl.load(positions_ptr + offs * 3 + 2, mask=live, other=0.0)

    dx = px - cam_x
    dy = py - cam_y
    dz = pz - cam_z
    d2 = dx * dx + dy * dy + dz * dz
    dist = tl.sqrt(d2)
    live = live & (dist > 1e-6)
    if radius_sq > 0.0:
        live = live & (d2 <= radius_sq)
    inv_dist = tl.where(dist > 1e-6, 1.0 / dist, 0.0)
    ux = dx * inv_dist
    uy = dy * inv_dist
    uz = dz * inv_dist

    inv_voxel = 1.0 / voxel_size
    inv_trunc = 1.0 / trunc

    for s in tl.static_range(-STEPS, STEPS + 1):
        t = s * voxel_size
        vx = tl.floor((px + ux * t) * inv_voxel).to(tl.int32)
        vy = tl.floor((py + uy * t) * inv_voxel).to(tl.int32)
        vz = tl.floor((pz + uz * t) * inv_voxel).to(tl.int32)

        cx = (vx.to(tl.float32) + 0.5) * voxel_size
        cy = (vy.to(tl.float32) + 0.5) * voxel_size
        cz = (vz.to(tl.float32) + 0.5) * voxel_size
        ex = cx - cam_x
        ey = cy - cam_y
        ez = cz - cam_z
        sdf = dist - tl.sqrt(ex * ex + ey * ey + ez * ez)
        active = live & (sdf >= -trunc)

        bx = _floor_div(vx, BLOCK_DIM)
        by = _floor_div(vy, BLOCK_DIM)
        bz = _floor_div(vz, BLOCK_DIM)
        want = _pack(bx, by, bz)
        start = _hash(bx, by, bz, hash_mask)

        # Lookup. Fixed trip count with a done mask: no per-lane early exit.
        found = tl.zeros((BLOCK,), tl.int32) - 1
        done = ~active
        for p in tl.static_range(0, MAX_PROBE):
            slot = (start + p) & hash_mask
            key = tl.load(table_i64 + slot * 2)
            hit = (key == want) & (~done)
            idx = tl.load(table_i32 + slot * 4 + 2)
            found = tl.where(hit, idx, found)
            done = done | hit
            # An empty slot means the block was never allocated: stop looking.
            done = done | ((key == EMPTY_KEY) & (~done))

        ok = active & (found >= 0)
        lx = vx - bx * BLOCK_DIM
        ly = vy - by * BLOCK_DIM
        lz = vz - bz * BLOCK_DIM
        vidx = found * BLOCK_VOXELS + (lz * BLOCK_DIM + ly) * BLOCK_DIM + lx

        # Approximate weight cap, read plainly, matching the other arms.
        # Enforcing it exactly would need a read-modify-write, which is what the
        # weighted-sum form exists to avoid.
        w_now = tl.load(weight_ptr + vidx, mask=ok, other=0.0)
        if weight_cap > 0.0:
            ok = ok & (w_now < weight_cap)

        sdf_n = tl.minimum(tl.maximum(sdf * inv_trunc, -1.0), 1.0)
        # Weighted SUMS, not running means. Sums commute, so plain atomic adds
        # suffice and no read-modify-write exists to race.
        tl.atomic_add(weight_ptr + vidx, 1.0, mask=ok)
        tl.atomic_add(tsdf_ptr + vidx, sdf_n, mask=ok)
