"""Spike S2 for the tsdf-backends project.

Question: can Triton (@triton.jit) express the two primitives the hash TSDF
integrate path needs, and how much does it cost in expressiveness?

Deliberately mirrors spike S1 (cuda-oxide) test-for-test so the results are
directly comparable, and mirrors the real kernel in
openstrate-reconstruct-rs/vendor/libinfer/src/tsdf_hash.cu:
  - hash_integrate_kernel : atomicAdd accumulation into an SoA voxel pool
  - line 338              : atomicCAS linear-probe hash insertion

Test A (regular)   : SoA atomic_add accumulation. Expected to be Triton's
                     natural ground.
Test B (irregular) : atomicCAS linear-probe insertion. This is the one the
                     roadmap flags as possibly infeasible.

Known API asymmetry found before running: tl.atomic_add accepts mask=, but
tl.atomic_cas does NOT. Lanes that have already resolved therefore cannot be
masked off, so they are redirected to a scratch slot holding a sentinel that can
never satisfy the CAS. Slot HASH_SIZE is that scratch.
"""

import torch
import triton
import triton.language as tl

HASH_SIZE = 1024          # power of two, mask-based hash as in the CUDA original
HASH_EMPTY = -1           # empty-slot sentinel
SCRATCH_SENTINEL = -2     # scratch slot value; must never equal HASH_EMPTY
N_COORDS = 64             # distinct block coords contended for
DUP = 8                   # threads per coord, i.e. contention factor
THREADS = N_COORDS * DUP  # 512, same as S1
MAX_PROBE = 64            # linear-probe bound, as in the CUDA version
BLOCK = 256


@triton.jit
def hash_insert_kernel(
    table_ptr, slot_out_ptr, won_out_ptr, n_threads,
    HASH_SIZE: tl.constexpr, HASH_EMPTY: tl.constexpr, DUP: tl.constexpr,
    MAX_PROBE: tl.constexpr, BLOCK: tl.constexpr,
):
    """atomicCAS linear-probe insert, mirroring tsdf_hash.cu:338."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_threads

    key = (offs // DUP).to(tl.int32)

    # Same cheap integer mix as the CUDA side, masked to table size.
    h = (key * 73856093) ^ (key * 19349663)
    h = h & (HASH_SIZE - 1)

    resolved = tl.full((BLOCK,), HASH_EMPTY, tl.int32)
    won = tl.zeros((BLOCK,), tl.int32)
    # tt.atomic_cas verifies that the cmp operand dtype matches the pointee
    # dtype exactly, so the sentinel must be a materialised int32 tensor rather
    # than a Python constexpr int.
    empty_i32 = tl.full((BLOCK,), HASH_EMPTY, tl.int32)
    # Out-of-range lanes start "done" so they never touch the table.
    done = ~valid

    # NOTE: fixed trip count. Triton has no per-lane early exit, so every
    # launch pays the full MAX_PROBE cost even when all lanes resolve on the
    # first probe. The CUDA original breaks out of the loop per thread.
    for p in range(MAX_PROBE):
        idx = (h + p) & (HASH_SIZE - 1)

        # Redirect resolved lanes to the scratch slot so their loads and CAS
        # attempts are inert. This is the workaround for atomic_cas lacking mask.
        eff_load = tl.where(done, HASH_SIZE, idx)
        observed = tl.load(table_ptr + eff_load)

        # Case 1: a peer already inserted our key here, so adopt the slot.
        hit = (observed == key) & (~done)
        resolved = tl.where(hit, idx, resolved)
        done = done | hit

        # Case 2: slot looks empty, so try to claim it.
        empty = (observed == HASH_EMPTY) & (~done)
        eff_cas = tl.where(empty, idx, HASH_SIZE)
        old = tl.atomic_cas(table_ptr + eff_cas, empty_i32, key)

        gotit = empty & (old == HASH_EMPTY)
        resolved = tl.where(gotit, idx, resolved)
        won = tl.where(gotit, 1, won)
        done = done | gotit

        # Lost the race but the winner wrote our key, so share the slot.
        adopt = empty & (old == key)
        resolved = tl.where(adopt, idx, resolved)
        done = done | adopt
        # Otherwise a different key took the slot: keep probing.

    tl.store(slot_out_ptr + offs, resolved, mask=valid)
    tl.store(won_out_ptr + offs, won, mask=valid)


@triton.jit
def voxel_accumulate_kernel(
    slot_ptr, sum_xyz_ptr, weight_ptr, count_ptr, n_threads,
    HASH_EMPTY: tl.constexpr, BLOCK: tl.constexpr,
):
    """Per-voxel SoA accumulation, mirroring hash_integrate_kernel."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_threads

    slot = tl.load(slot_ptr + offs, mask=valid, other=HASH_EMPTY)
    live = valid & (slot != HASH_EMPTY)

    # atomic_add accepts a mask, so no scratch-slot workaround is needed here.
    tl.atomic_add(sum_xyz_ptr + slot * 3 + 0, 1.0, mask=live)
    tl.atomic_add(sum_xyz_ptr + slot * 3 + 1, 2.0, mask=live)
    tl.atomic_add(sum_xyz_ptr + slot * 3 + 2, 3.0, mask=live)
    tl.atomic_add(weight_ptr + slot, 0.5, mask=live)
    tl.atomic_add(count_ptr + slot, 1, mask=live)


def main():
    dev = "cuda"
    print("=== TSDF hash-insert + accumulate spike (Triton) ===")
    print(f"triton {triton.__version__}  device {torch.cuda.get_device_name(0)} "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}\n")

    grid = (triton.cdiv(THREADS, BLOCK),)
    ok = True

    # -- Test B: atomicCAS linear-probe insertion ---------------------------
    print("--- Test B: atomicCAS linear-probe hash insert (the irregular part) ---")
    print(f"  {THREADS} threads racing for {N_COORDS} distinct coords ({DUP}x contention)")
    # Slot HASH_SIZE is the scratch sink; it must never satisfy the CAS.
    table = torch.full((HASH_SIZE + 1,), HASH_EMPTY, dtype=torch.int32, device=dev)
    table[HASH_SIZE] = SCRATCH_SENTINEL
    slot = torch.zeros(THREADS, dtype=torch.int32, device=dev)
    won = torch.zeros(THREADS, dtype=torch.int32, device=dev)

    hash_insert_kernel[grid](
        table, slot, won, THREADS,
        HASH_SIZE=HASH_SIZE, HASH_EMPTY=HASH_EMPTY, DUP=DUP,
        MAX_PROBE=MAX_PROBE, BLOCK=BLOCK,
    )
    torch.cuda.synchronize()

    slots = slot.cpu()
    winners = int(won.sum())
    unresolved = int((slots == HASH_EMPTY).sum())
    print(f"  winners = {winners} (expected {N_COORDS})")
    print(f"  unresolved threads = {unresolved} (expected 0)")

    base = slots.view(N_COORDS, DUP)[:, :1]
    agree = bool((slots.view(N_COORDS, DUP) == base).all())
    print(f"  all duplicates agree on slot = {agree}")
    distinct = int(torch.unique(base).numel())
    print(f"  distinct slots used = {distinct} (expected {N_COORDS})")
    # Scratch must be untouched by any successful claim.
    scratch_intact = int(table[HASH_SIZE]) == SCRATCH_SENTINEL
    print(f"  scratch slot intact = {scratch_intact}")

    tB = (winners == N_COORDS and unresolved == 0 and agree
          and distinct == N_COORDS and scratch_intact)
    print(f"  => Test B {'PASS' if tB else 'FAIL'}\n")
    ok &= tB

    # -- Test A: SoA atomicAdd accumulation ---------------------------------
    print("--- Test A: atomicAdd SoA voxel accumulation (the regular part) ---")
    sum_xyz = torch.zeros(HASH_SIZE * 3, dtype=torch.float32, device=dev)
    weight = torch.zeros(HASH_SIZE, dtype=torch.float32, device=dev)
    count = torch.zeros(HASH_SIZE, dtype=torch.int32, device=dev)

    voxel_accumulate_kernel[grid](
        slot, sum_xyz, weight, count, THREADS,
        HASH_EMPTY=HASH_EMPTY, BLOCK=BLOCK,
    )
    torch.cuda.synchronize()

    total_count = int(count.sum())
    total_w = float(weight.sum())
    total_x = float(sum_xyz[0::3].sum())
    total_z = float(sum_xyz[2::3].sum())
    print(f"  total count  = {total_count} (expected {THREADS})")
    print(f"  total weight = {total_w} (expected {THREADS * 0.5})")
    print(f"  total sum_x  = {total_x} (expected {float(THREADS)})")
    print(f"  total sum_z  = {total_z} (expected {THREADS * 3.0})")
    per_slot_ok = bool((count.cpu()[base.flatten().long()] == DUP).all())
    print(f"  every occupied slot has exactly {DUP} observations = {per_slot_ok}")

    tA = (total_count == THREADS and abs(total_w - THREADS * 0.5) < 1e-3
          and abs(total_x - THREADS) < 1e-2
          and abs(total_z - THREADS * 3.0) < 1e-2 and per_slot_ok)
    print(f"  => Test A {'PASS' if tA else 'FAIL'}\n")
    ok &= tA

    print(f"=== SPIKE RESULT: {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
