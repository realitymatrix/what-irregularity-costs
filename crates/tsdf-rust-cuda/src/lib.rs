//! Arm A4: the TSDF integrate path written in Rust, compiled to PTX by
//! `cuda-oxide`.
//!
//! Deliberately a line-by-line counterpart of the CUDA C++ arm (A3), not an
//! idiomatic-Rust reinterpretation: same packed 64-bit hash key, same
//! open-addressed table with linear probing, same compare-exchange insertion,
//! same truncation-band walk, same voxel-centre signed distance, same
//! weighted-sum accumulation. A difference in results would then be a
//! difference in code generation, which is the thing being measured. NVIDIA
//! has published no such comparison; their own `cuda-oxide` vs CUDA C++
//! appendix is an empty placeholder.
//!
//! # Scope: arms differ in the INTEGRATE path, not in extraction
//!
//! A3, A4, A5a and A5b each implement `allocate_blocks` and `update_voxels`;
//! surface extraction is shared. Two reasons:
//!
//!   * The claim under test is about fusion throughput. Extraction runs once
//!     per sequence, integrate runs once per frame.
//!   * Three independent marching-tetrahedra implementations would differ in
//!     ways that have nothing to do with the languages, and those differences
//!     would land in the same numbers as the effect being measured.
//!
//! A2 (CPU) necessarily has its own extractor, but it is on a different device
//! class and is reported on its own axis anyway.
//!
//! Build (needs the pinned nightly and the cuda-oxide backend):
//!   cargo oxide run --arch sm_120 --materialize-cubin

use cuda_core::{CudaContext, LaunchConfig};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32, DeviceAtomicI32, DeviceAtomicI64};
use cuda_device::fence::threadfence;
use cuda_device::{kernel, ptx_asm, thread};
use cuda_host::cuda_module;

/// Must match `osn_tsdf::kBlockDim`. 8^3 = 512 voxels per block.
const BLOCK_DIM: i32 = 8;
const BLOCK_VOXELS: i32 = BLOCK_DIM * BLOCK_DIM * BLOCK_DIM;
/// Empty-slot sentinel. Cannot collide with a packed key, which is biased
/// non-negative.
const EMPTY_KEY: i64 = -1;
const COORD_BIAS: i64 = 1 << 20;
const COORD_BITS: i64 = 21;

#[cuda_module]
mod kernels {
    use super::*;

    /// Pack a block coordinate into one 64-bit key.
    ///
    /// The entire coordinate must live in one word so a single
    /// compare-exchange publishes all of it. Splitting it lets a reader match
    /// part of a slot, mismatch on the rest, and probe onward; at GPU thread
    /// counts that degenerates into full-table scans. The C++ arms hit exactly
    /// that before the layout was changed.
    #[inline(always)]
    fn pack(x: i32, y: i32, z: i32) -> i64 {
        (((x as i64) + COORD_BIAS) << (2 * COORD_BITS))
            | (((y as i64) + COORD_BIAS) << COORD_BITS)
            | ((z as i64) + COORD_BIAS)
    }

    /// Same multiply-xor mix as the C++ arms. Must stay bit-identical: a
    /// different mix changes probe sequences, which changes which blocks share
    /// cache lines, and that would surface as a "language" difference.
    #[inline(always)]
    fn hash_block(x: i32, y: i32, z: i32, mask: u32) -> u32 {
        let h = (x as u32).wrapping_mul(73856093)
            ^ (y as u32).wrapping_mul(19349663)
            ^ (z as u32).wrapping_mul(83492791);
        h & mask
    }

    /// floor(a / b) for positive b. Plain integer division truncates toward
    /// zero, which mirrors block coordinates across the origin and puts a seam
    /// at exactly x = 0.
    #[inline(always)]
    fn floor_div(a: i32, b: i32) -> i32 {
        let q = a / b;
        if a % b != 0 && ((a < 0) != (b < 0)) { q - 1 } else { q }
    }

    /// Look up a block, or -1.
    ///
    /// # Safety
    /// `table` must hold `mask + 1` entries laid out as (i64 key, i32 idx, i32 pad).
    #[inline(always)]
    unsafe fn find_block(table: *mut i64, mask: u32, bx: i32, by: i32, bz: i32) -> i32 {
        // SAFETY: caller guarantees the table and pool spans; see the
        // doc comment. Edition 2024 requires the block to be explicit.
        unsafe {
            let want = pack(bx, by, bz);
            let size = mask + 1;
            let mut probe: u32 = 0;
            let start = hash_block(bx, by, bz, mask);
            while probe < size {
                let slot = ((start + probe) & mask) as usize;
                // Entry is 16 bytes: key at +0, block_idx at +8.
                let key_ptr = table.add(slot * 2);
                let idx_ptr = key_ptr.add(1) as *mut i32;
                // PLAIN read, not volatile. The probe visits a different slot each
            // iteration, so nothing can be hoisted, and a volatile load forces
            // an uncached access on the hottest path in the kernel. A3 reads
            // the key plainly for exactly this reason and reserves volatile for
            // the block_idx spin below, which does re-read one address and
            // therefore must observe another thread's publication.
            //
            // Measured: making this volatile cost the allocate stage 1.64x
            // against A3 at identical SASS instruction counts, which is what
            // sent the attribution hunting for occupancy and instruction mix
            // before the asymmetry was spotted.
            let key = key_ptr.read();
                if key == EMPTY_KEY {
                    return -1;
                }
                if key == want {
                    // Spin until the winner publishes the index; seeing the key
                    // without the index would read -1 and drop the point.
                    // Scoped atomic load, GPU scope, not `read_volatile`.
                    // Both re-read the address every iteration, but
                    // `read_volatile` is a Rust volatile access and lowers to a
                    // SYSTEM-scope strongly-ordered load, ordering against the
                    // host and peer devices. Nothing here needs that: the table
                    // lives in one device's global memory. See docs/A3-A4-GAP.md.
                    let idx_atomic = DeviceAtomicI32::from_ptr(idx_ptr);
                    let mut idx = idx_atomic.load(AtomicOrdering::Relaxed);
                    while idx < 0 {
                        idx = idx_atomic.load(AtomicOrdering::Relaxed);
                    }
                    return idx;
                }
                probe += 1;
            }
            -1
    
        }
}

    /// Find or insert. Returns -1 when the pool is exhausted.
    #[inline(always)]
    /// MEASUREMENT ONLY beyond the production instantiation.
    ///
    /// `CAS`, `SPIN` and `COUNT` are const so each variant specialises rather
    /// than branching inside the probe loop; a runtime flag would change the
    /// codegen being measured. The production kernel instantiates
    /// `<true, true, true>`, which is byte-identical to the previous
    /// non-generic function. Any other instantiation is deliberately WRONG and
    /// exists only to price one component. See docs/A3-A4-GAP.md.
    unsafe fn find_or_insert<const CAS: bool, const SPIN: bool, const COUNT: bool,
                             const FENCE: bool, const PUBLISH: bool,
                             const COUNTCAS: bool>(
        table: *mut i64,
        mask: u32,
        counters: *mut i32,
        pool_capacity: i32,
        drop_count: *mut i64,
        block_coord: *mut i32,
        bx: i32,
        by: i32,
        bz: i32,
    ) -> i32 {
        // SAFETY: caller guarantees the table and pool spans; see the
        // doc comment. Edition 2024 requires the block to be explicit.
        unsafe {
            let want = pack(bx, by, bz);
            let size = mask + 1;
            let start = hash_block(bx, by, bz, mask);
            let mut probe: u32 = 0;
            while probe < size {
                let slot = ((start + probe) & mask) as usize;
                let key_ptr = table.add(slot * 2);
                let idx_ptr = key_ptr.add(1) as *mut i32;
                let key_atomic = DeviceAtomicI64::from_ptr(key_ptr);
                // PLAIN read, not an atomic load, and this is the hot path: it
                // runs on every probe iteration. An atomic load at GPU scope
                // must be coherent across SMs, so it cannot be served from L1;
                // profiling showed that pushing 1.70x the sectors to L2 against
                // A3 (964k against 569k) for an identical request count, which
                // is where the extra long_scoreboard stalls come from.
                //
                // Correct because the probe visits a DIFFERENT slot each
                // iteration, so nothing can be hoisted, and a stale key only
                // costs an extra probe. A3 reads its key plainly for the same
                // reason. The block_idx spin below still needs a real atomic
                // load: it re-reads ONE address until a peer publishes.
                let key = key_ptr.read();

                if key == want {
                    // Scoped atomic load, GPU scope, not `read_volatile`.
                    // Both re-read the address every iteration, but
                    // `read_volatile` is a Rust volatile access and lowers to a
                    // SYSTEM-scope strongly-ordered load, ordering against the
                    // host and peer devices. Nothing here needs that: the table
                    // lives in one device's global memory. See docs/A3-A4-GAP.md.
                    let idx_atomic = DeviceAtomicI32::from_ptr(idx_ptr);
                    let mut idx = idx_atomic.load(AtomicOrdering::Relaxed);
                    if SPIN {
                        while idx < 0 {
                            idx = idx_atomic.load(AtomicOrdering::Relaxed);
                        }
                    }
                    return idx;
                }
                if key == EMPTY_KEY {
                    if !CAS {
                        // Price the compare-exchange: probe and read, never insert.
                        return -1;
                    }
                    if COUNTCAS {
                        // Tally every compare-exchange ATTEMPT. Arm A3 carries
                        // the identical switch, so the counts are directly
                        // comparable. Overloading drop_count is safe because
                        // the counted kernel only runs on a pool large enough
                        // that nothing is dropped.
                        DeviceAtomicI64::from_ptr(drop_count).fetch_add(1, AtomicOrdering::Relaxed);
                    }
                    match key_atomic.compare_exchange(
                        EMPTY_KEY,
                        want,
                        AtomicOrdering::Relaxed,
                    AtomicOrdering::Relaxed,
                    ) {
                        Ok(_) => {
                            let idx = if COUNT {
                                DeviceAtomicI32::from_ptr(counters)
                                    .fetch_add(1, AtomicOrdering::Relaxed)
                            } else {
                                // Price the counter contention: a fixed slot.
                                // Wrong, and only ever launched by the probe.
                                (bx & 0x3ff).abs()
                            };
                            if idx >= pool_capacity {
                                DeviceAtomicI32::from_ptr(counters)
                                    .fetch_sub(1, AtomicOrdering::Relaxed);
                                // Atomic exchange, not a store: libNVVM allows RMW
                            // but not atomic stores. Relaxed, because a Release
                            // ordering lowers to an LLVM `fence`, which libNVVM
                            // also rejects ("Illegal instruction: fence"). The
                            // ordering comes from an explicit threadfence, the
                            // same construct A3 uses.
                            if FENCE { threadfence(); }
                            key_atomic.swap(EMPTY_KEY, AtomicOrdering::Relaxed);
                                DeviceAtomicI64::from_ptr(drop_count)
                                    .fetch_add(1, AtomicOrdering::Relaxed);
                                return -1;
                            }
                            if PUBLISH {
                                let bc = block_coord.add((idx * 3) as usize);
                                bc.write(bx);
                                bc.add(1).write(by);
                                bc.add(2).write(bz);
                            }
                            // Publish the index only after the coordinate writes
                        // are visible. Explicit threadfence + relaxed exchange,
                        // exactly what A3 does (__threadfence + atomicExch):
                        // a Release-ordered RMW would lower to an LLVM fence,
                        // which libNVVM rejects.
                        if PUBLISH {
                            if FENCE { threadfence(); }
                            DeviceAtomicI32::from_ptr(idx_ptr).swap(idx, AtomicOrdering::Relaxed);
                        }
                            return idx;
                        }
                        Err(actual) => {
                            if actual == want {
                                let idx_atomic = DeviceAtomicI32::from_ptr(idx_ptr);
                                let mut idx = idx_atomic.load(AtomicOrdering::Relaxed);
                                if SPIN {
                                    while idx < 0 {
                                        idx = idx_atomic.load(AtomicOrdering::Relaxed);
                                    }
                                }
                                return idx;
                            }
                        }
                    }
                }
                probe += 1;
            }
            DeviceAtomicI64::from_ptr(drop_count).fetch_add(1, AtomicOrdering::Relaxed);
            -1
    
        }
}

    /// Find or insert, publishing the key and the block index in ONE 128-bit
    /// compare-exchange.
    ///
    /// The spin in `find_or_insert` exists to cover a window: the key is
    /// published by a 64-bit CAS and the index by a later store, so a reader can
    /// see a key whose index is still -1 and must wait. Measured, that wait is
    /// 75% to 83% of the whole A4/A3 allocate gap at realistic point counts
    /// (docs/A3-A4-GAP.md). Publishing both words atomically removes the window
    /// rather than making the wait cheaper, so a visible key implies a visible
    /// index and no reader ever waits.
    ///
    /// DOES NOT WORK, and the reason is the point of keeping it.
    ///
    /// The pool index must be reserved BEFORE the exchange, since it is part of
    /// the value being written, so a thread that loses the slot holds an index
    /// it cannot use. The intended mitigation was to reserve only when a slot is
    /// actually observed empty, on the assumption that once a block exists later
    /// threads match its key and never reserve.
    ///
    /// That assumption fails at exactly the contention this is meant to fix. At
    /// kernel start the table is empty and every thread targeting a given block
    /// observes an empty slot before any exchange lands, so they all reserve.
    /// Measured at 320k points: 65,536 reservations (the whole pool) against
    /// 1,160 real blocks, and 2.3M dropped points. The best-effort give-back,
    /// a compare-exchange returning the counter from `claimed + 1` to `claimed`,
    /// only succeeds for the last reserver and so recovers almost nothing.
    ///
    /// The 128-bit instruction itself is fine: `bench/probe_cas128_selftest.cu`
    /// exercises `atom.global.acq_rel.gpu.cas.b128` in isolation and it matches
    /// and writes correctly. Substituting a plain 64-bit compare-exchange here
    /// while keeping the reserve-first structure saturates the pool identically,
    /// which is what isolates the fault to the algorithm.
    ///
    /// Removing the spin therefore needs an index that does not have to be
    /// reserved in advance: deriving the block index from the hash slot would
    /// do it, at the cost of sizing the voxel pool to the table rather than to
    /// the block count.
    ///
    /// Entry layout is (i64 key, i32 block_idx, i32 pad), so the low 64 bits are
    /// the key and the high 64 bits are the index in their low half. Empty is
    /// (-1, -1, 0), hence an expected high word of 0x0000_0000_FFFF_FFFF.
    #[allow(clippy::too_many_arguments)]
    unsafe fn find_or_insert_128(
        table: *mut i64,
        mask: u32,
        counters: *mut i32,
        pool_capacity: i32,
        drop_count: *mut i64,
        block_coord: *mut i32,
        bx: i32,
        by: i32,
        bz: i32,
    ) -> i32 {
        unsafe {
            let want = pack(bx, by, bz);
            let size = mask + 1;
            let start = hash_block(bx, by, bz, mask);
            let mut probe: u32 = 0;
            // Reserved lazily, kept across probes so a collision on a different
            // key does not reserve twice.
            let mut claimed: i32 = -1;

            while probe < size {
                let slot = ((start + probe) & mask) as usize;
                let key_ptr = table.add(slot * 2);
                let idx_ptr = key_ptr.add(1) as *mut i32;

                // Acquire: pairs with the release in the exchange below, so
                // seeing the key guarantees seeing the index written with it.
                let key = DeviceAtomicI64::from_ptr(key_ptr).load(AtomicOrdering::Acquire);

                if key == want {
                    return DeviceAtomicI32::from_ptr(idx_ptr).load(AtomicOrdering::Relaxed);
                }

                if key == EMPTY_KEY {
                    if claimed < 0 {
                        let idx = DeviceAtomicI32::from_ptr(counters)
                            .fetch_add(1, AtomicOrdering::Relaxed);
                        if idx >= pool_capacity {
                            DeviceAtomicI32::from_ptr(counters)
                                .fetch_sub(1, AtomicOrdering::Relaxed);
                            DeviceAtomicI64::from_ptr(drop_count)
                                .fetch_add(1, AtomicOrdering::Relaxed);
                            return -1;
                        }
                        // Safe before publication: the index is unreachable
                        // until the exchange makes it visible.
                        let bc = block_coord.add((idx * 3) as usize);
                        bc.write(bx);
                        bc.add(1).write(by);
                        bc.add(2).write(bz);
                        claimed = idx;
                    }

                    let exp_lo: u64 = EMPTY_KEY as u64;
                    let exp_hi: u64 = 0x0000_0000_FFFF_FFFF;
                    let des_lo: u64 = want as u64;
                    let des_hi: u64 = claimed as u32 as u64;
                    let mut old_lo: u64 = 0;
                    let mut old_hi: u64 = 0;
                    // `atom.cas.b128` takes .b128 register operands, not a pair
                    // of .b64s, so the value is packed into a .b128 temporary
                    // with `mov.b128` and unpacked the same way. Every inline
                    // operand therefore stays 64-bit.
                    ptx_asm!(
                        // `cvta.to.global` first: the pointer is generic, and
                        // passing a generic address to a `.global` atomic is
                        // undefined. Without it the exchange never matches, the
                        // pool saturates and every point is dropped.
                        "{ .reg .b128 t, e, d;\n\t\
                           .reg .u64 g;\n\t\
                           cvta.to.global.u64 g, %6;\n\t\
                           mov.b128 e, {%2, %3};\n\t\
                           mov.b128 d, {%4, %5};\n\t\
                           atom.global.acq_rel.gpu.cas.b128 t, [g], e, d;\n\t\
                           mov.b128 {%0, %1}, t; }",
                        out("=l") old_lo,
                        out("=l") old_hi,
                        in("l") exp_lo,
                        in("l") exp_hi,
                        in("l") des_lo,
                        in("l") des_hi,
                        in("l") key_ptr as u64,
                        clobber("memory")
                    );

                    if old_lo == exp_lo {
                        return claimed;
                    }
                    if old_lo as i64 == want {
                        // Lost to a peer inserting the same block. Its index is
                        // already in the value we read back, so still no wait.
                        // `claimed` is now spare; best-effort give-back only
                        // succeeds if nothing else has reserved since.
                        DeviceAtomicI32::from_ptr(counters)
                            .compare_exchange(
                                claimed + 1,
                                claimed,
                                AtomicOrdering::Relaxed,
                                AtomicOrdering::Relaxed,
                            )
                            .ok();
                        return (old_hi & 0xFFFF_FFFF) as i32;
                    }
                }
                probe += 1;
            }
            DeviceAtomicI64::from_ptr(drop_count).fetch_add(1, AtomicOrdering::Relaxed);
            -1
        }
    }

    /// Self-test for `atom.cas.b128` through `ptx_asm!`.
    ///
    /// One thread, one 16-byte cell preset to (-1, -1, 0), one exchange to
    /// (42, 7). Writes back the value the exchange returned and the resulting
    /// memory, so the host can see whether the instruction ran at all. Exists
    /// because the 128-bit insert saturated the pool, which is what a CAS that
    /// never matches looks like from the outside.
    #[kernel]
    pub fn cas128_selftest(cell: &[i64], out: &[i64]) {
        let tid = thread::threadIdx_x();
        if tid != 0 {
            return;
        }
        unsafe {
            let p = cell.as_ptr() as *mut i64;
            let exp_lo: u64 = (-1i64) as u64;
            let exp_hi: u64 = 0x0000_0000_FFFF_FFFF;
            let des_lo: u64 = 42;
            let des_hi: u64 = 7;
            let mut old_lo: u64 = 0xDEAD;
            let mut old_hi: u64 = 0xBEEF;
            ptx_asm!(
                "{ .reg .b128 t, e, d;\n\t\
                   .reg .u64 g;\n\t\
                   cvta.to.global.u64 g, %6;\n\t\
                   mov.b128 e, {%2, %3};\n\t\
                   mov.b128 d, {%4, %5};\n\t\
                   atom.global.acq_rel.gpu.cas.b128 t, [g], e, d;\n\t\
                   mov.b128 {%0, %1}, t; }",
                out("=l") old_lo,
                out("=l") old_hi,
                in("l") exp_lo,
                in("l") exp_hi,
                in("l") des_lo,
                in("l") des_hi,
                in("l") p as u64,
                clobber("memory")
            );
            let o = out.as_ptr() as *mut i64;
            o.write(old_lo as i64);
            o.add(1).write(old_hi as i64);
            o.add(2).write(p.read());
            o.add(3).write(p.add(1).read());
        }
    }

    /// Find or insert with the block index DERIVED FROM THE SLOT.
    ///
    /// The spin exists because a hash entry holds two logically different
    /// things: the block's identity, known before insertion, and its location
    /// in the voxel pool, assigned by a counter and known only after winning
    /// the slot. Publishing both atomically needs the location first;
    /// the location is only yours if the exchange succeeds. That circularity is
    /// the whole problem, and it has exactly two escapes: publish identity then
    /// location and make readers wait (what both arms do), or reserve the
    /// location speculatively and leak it under contention (tried, and it
    /// saturates the pool).
    ///
    /// Setting `block_idx = slot` removes the circularity instead of working
    /// around it. There is then only one value to publish, a 64-bit exchange
    /// suffices, and a reader that matches a key knows the index immediately.
    /// No counter, no publication store, no fence, and no wait.
    ///
    /// The price is that the voxel pool must have an entry per table slot. Here
    /// that is arranged without changing any allocation by masking the hash to
    /// `pool_capacity` rather than to `hash_mask`, so the effective table is the
    /// pool. Since the allocated table is twice the pool, half of it goes
    /// unused and the effective load factor doubles. A real implementation would
    /// size the pool to the table instead and pay roughly 2x the voxel memory.
    ///
    /// `counters` is still incremented once per successful insert, but only so
    /// the host can check the block count. Nothing reads it, and no thread waits
    /// on it.
    #[allow(clippy::too_many_arguments)]
    unsafe fn find_or_insert_slotidx(
        table: *mut i64,
        pool_capacity: i32,
        counters: *mut i32,
        drop_count: *mut i64,
        block_coord: *mut i32,
        bx: i32,
        by: i32,
        bz: i32,
    ) -> i32 {
        unsafe {
            // Effective table = the pool, so every slot has voxel storage.
            let mask = (pool_capacity as u32).wrapping_sub(1);
            let want = pack(bx, by, bz);
            let start = hash_block(bx, by, bz, mask);
            let mut probe: u32 = 0;
            while probe <= mask {
                let slot = ((start + probe) & mask) as usize;
                let key_ptr = table.add(slot * 2);
                let key_atomic = DeviceAtomicI64::from_ptr(key_ptr);
                let key = key_atomic.load(AtomicOrdering::Relaxed);

                // A matching key means the block is ours to use, and its index
                // IS this slot. Nothing to wait for.
                if key == want {
                    return slot as i32;
                }
                if key == EMPTY_KEY {
                    match key_atomic.compare_exchange(
                        EMPTY_KEY,
                        want,
                        AtomicOrdering::Relaxed,
                        AtomicOrdering::Relaxed,
                    ) {
                        Ok(_) => {
                            let bc = block_coord.add((slot * 3) as usize);
                            bc.write(bx);
                            bc.add(1).write(by);
                            bc.add(2).write(bz);
                            DeviceAtomicI32::from_ptr(counters)
                                .fetch_add(1, AtomicOrdering::Relaxed);
                            return slot as i32;
                        }
                        // Lost the race for this slot. If the winner wanted the
                        // same block, the slot is still the answer.
                        Err(actual) if actual == want => return slot as i32,
                        Err(_) => {}
                    }
                }
                probe += 1;
            }
            DeviceAtomicI64::from_ptr(drop_count).fetch_add(1, AtomicOrdering::Relaxed);
            -1
        }
    }

    /// Pass 1: allocate every block the batch's truncation band touches.
    #[kernel]
    pub fn alloc_kernel(
        positions: &[f32],
        table: &[i64],
        counters: &[i32],
        block_coord: &[i32],
        drop_count: &[i64],
        n_points: i32,
        pool_capacity: i32,
        hash_mask: u32,
        voxel_size: f32,
        trunc: f32,
        cam_x: f32,
        cam_y: f32,
        cam_z: f32,
        radius_sq: f32,
    ) {
        let i = thread::index_1d().get() as i32;
        if i >= n_points {
            return;
        }
        unsafe {
            let p = positions.as_ptr().add((i * 3) as usize);
            let (px, py, pz) = (*p, *p.add(1), *p.add(2));
            let (dx, dy, dz) = (px - cam_x, py - cam_y, pz - cam_z);
            let d2 = dx * dx + dy * dy + dz * dz;
            if radius_sq > 0.0 && d2 > radius_sq {
                return;
            }
            let dist = d2.sqrt();
            if !(dist > 1e-6) {
                return;
            }
            let (ux, uy, uz) = (dx / dist, dy / dist, dz / dist);
            let inv_voxel = 1.0f32 / voxel_size;
            let steps = ((trunc * inv_voxel).ceil() as i32).max(1);

            let mut s = -steps;
            while s <= steps {
                let t = (s as f32) * voxel_size;
                let vx = ((px + ux * t) * inv_voxel).floor() as i32;
                let vy = ((py + uy * t) * inv_voxel).floor() as i32;
                let vz = ((pz + uz * t) * inv_voxel).floor() as i32;

                // Same occlusion cull as the update pass. Allocation must apply
                // exactly the same gate, or the pool fills with blocks that
                // never receive a contribution and the block count diverges
                // from the other arms while the meshes still match.
                let cx = ((vx as f32) + 0.5) * voxel_size;
                let cy = ((vy as f32) + 0.5) * voxel_size;
                let cz = ((vz as f32) + 0.5) * voxel_size;
                let (ex, ey, ez) = (cx - cam_x, cy - cam_y, cz - cam_z);
                if dist - (ex * ex + ey * ey + ez * ez).sqrt() >= -trunc {
                    find_or_insert::<true, true, true, true, true, false>(
                        table.as_ptr() as *mut i64,
                        hash_mask,
                        counters.as_ptr() as *mut i32,
                        pool_capacity,
                        drop_count.as_ptr() as *mut i64,
                        block_coord.as_ptr() as *mut i32,
                        floor_div(vx, BLOCK_DIM),
                        floor_div(vy, BLOCK_DIM),
                        floor_div(vz, BLOCK_DIM),
                    );
                }
                s += 1;
            }
        }
    }

    /// Pass 1 with the block index derived from the hash slot: no counter
    /// dependency, no publication store, no fence, and no wait. See
    /// `find_or_insert_slotidx`.
    #[kernel]
    pub fn alloc_kernel_slotidx(
        positions: &[f32],
        table: &[i64],
        counters: &[i32],
        block_coord: &[i32],
        drop_count: &[i64],
        n_points: i32,
        pool_capacity: i32,
        hash_mask: u32,
        voxel_size: f32,
        trunc: f32,
        cam_x: f32,
        cam_y: f32,
        cam_z: f32,
        radius_sq: f32,
    ) {
        let i = thread::index_1d().get() as i32;
        if i >= n_points {
            return;
        }
        unsafe {
            let p = positions.as_ptr().add((i * 3) as usize);
            let (px, py, pz) = (*p, *p.add(1), *p.add(2));
            let (dx, dy, dz) = (px - cam_x, py - cam_y, pz - cam_z);
            let d2 = dx * dx + dy * dy + dz * dz;
            if radius_sq > 0.0 && d2 > radius_sq {
                return;
            }
            let dist = d2.sqrt();
            if !(dist > 1e-6) {
                return;
            }
            let (ux, uy, uz) = (dx / dist, dy / dist, dz / dist);
            let inv_voxel = 1.0f32 / voxel_size;
            let steps = ((trunc * inv_voxel).ceil() as i32).max(1);

            let mut s = -steps;
            while s <= steps {
                let t = (s as f32) * voxel_size;
                let vx = ((px + ux * t) * inv_voxel).floor() as i32;
                let vy = ((py + uy * t) * inv_voxel).floor() as i32;
                let vz = ((pz + uz * t) * inv_voxel).floor() as i32;

                // Same occlusion cull as the update pass. Allocation must apply
                // exactly the same gate, or the pool fills with blocks that
                // never receive a contribution and the block count diverges
                // from the other arms while the meshes still match.
                let cx = ((vx as f32) + 0.5) * voxel_size;
                let cy = ((vy as f32) + 0.5) * voxel_size;
                let cz = ((vz as f32) + 0.5) * voxel_size;
                let (ex, ey, ez) = (cx - cam_x, cy - cam_y, cz - cam_z);
                if dist - (ex * ex + ey * ey + ez * ez).sqrt() >= -trunc {
                    find_or_insert_slotidx(
                        table.as_ptr() as *mut i64,
                        pool_capacity,
                        counters.as_ptr() as *mut i32,
                        drop_count.as_ptr() as *mut i64,
                        block_coord.as_ptr() as *mut i32,
                        floor_div(vx, BLOCK_DIM),
                        floor_div(vy, BLOCK_DIM),
                        floor_div(vz, BLOCK_DIM),
                    );
                }
                s += 1;
            }
        }
    }

    /// Pass 1, publishing key and index in one 128-bit exchange so no reader
    /// ever waits. See `find_or_insert_128`. Correct, but may report a block
    /// count above A3's if any reserved index is leaked.
    #[kernel]
    pub fn alloc_kernel_cas128(
        positions: &[f32],
        table: &[i64],
        counters: &[i32],
        block_coord: &[i32],
        drop_count: &[i64],
        n_points: i32,
        pool_capacity: i32,
        hash_mask: u32,
        voxel_size: f32,
        trunc: f32,
        cam_x: f32,
        cam_y: f32,
        cam_z: f32,
        radius_sq: f32,
    ) {
        let i = thread::index_1d().get() as i32;
        if i >= n_points {
            return;
        }
        unsafe {
            let p = positions.as_ptr().add((i * 3) as usize);
            let (px, py, pz) = (*p, *p.add(1), *p.add(2));
            let (dx, dy, dz) = (px - cam_x, py - cam_y, pz - cam_z);
            let d2 = dx * dx + dy * dy + dz * dz;
            if radius_sq > 0.0 && d2 > radius_sq {
                return;
            }
            let dist = d2.sqrt();
            if !(dist > 1e-6) {
                return;
            }
            let (ux, uy, uz) = (dx / dist, dy / dist, dz / dist);
            let inv_voxel = 1.0f32 / voxel_size;
            let steps = ((trunc * inv_voxel).ceil() as i32).max(1);

            let mut s = -steps;
            while s <= steps {
                let t = (s as f32) * voxel_size;
                let vx = ((px + ux * t) * inv_voxel).floor() as i32;
                let vy = ((py + uy * t) * inv_voxel).floor() as i32;
                let vz = ((pz + uz * t) * inv_voxel).floor() as i32;

                // Same occlusion cull as the update pass. Allocation must apply
                // exactly the same gate, or the pool fills with blocks that
                // never receive a contribution and the block count diverges
                // from the other arms while the meshes still match.
                let cx = ((vx as f32) + 0.5) * voxel_size;
                let cy = ((vy as f32) + 0.5) * voxel_size;
                let cz = ((vz as f32) + 0.5) * voxel_size;
                let (ex, ey, ez) = (cx - cam_x, cy - cam_y, cz - cam_z);
                if dist - (ex * ex + ey * ey + ez * ez).sqrt() >= -trunc {
                    find_or_insert_128(
                        table.as_ptr() as *mut i64,
                        hash_mask,
                        counters.as_ptr() as *mut i32,
                        pool_capacity,
                        drop_count.as_ptr() as *mut i64,
                        block_coord.as_ptr() as *mut i32,
                        floor_div(vx, BLOCK_DIM),
                        floor_div(vy, BLOCK_DIM),
                        floor_div(vz, BLOCK_DIM),
                    );
                }
                s += 1;
            }
        }
    }

    /// MEASUREMENT ONLY, and deliberately incorrect. Insert normally but never wait for a peer to publish `block_idx`. Returns a
    /// stale -1 for a block another thread is mid-inserting, so the block count
    /// is wrong. Prices the publication spin.
    ///
    /// Exists to bisect the A4/A3 allocate gap; see docs/A3-A4-GAP.md.
    #[kernel]
    pub fn alloc_kernel_nospin(
        positions: &[f32],
        table: &[i64],
        counters: &[i32],
        block_coord: &[i32],
        drop_count: &[i64],
        n_points: i32,
        pool_capacity: i32,
        hash_mask: u32,
        voxel_size: f32,
        trunc: f32,
        cam_x: f32,
        cam_y: f32,
        cam_z: f32,
        radius_sq: f32,
    ) {
        let i = thread::index_1d().get() as i32;
        if i >= n_points {
            return;
        }
        unsafe {
            let p = positions.as_ptr().add((i * 3) as usize);
            let (px, py, pz) = (*p, *p.add(1), *p.add(2));
            let (dx, dy, dz) = (px - cam_x, py - cam_y, pz - cam_z);
            let d2 = dx * dx + dy * dy + dz * dz;
            if radius_sq > 0.0 && d2 > radius_sq {
                return;
            }
            let dist = d2.sqrt();
            if !(dist > 1e-6) {
                return;
            }
            let (ux, uy, uz) = (dx / dist, dy / dist, dz / dist);
            let inv_voxel = 1.0f32 / voxel_size;
            let steps = ((trunc * inv_voxel).ceil() as i32).max(1);

            let mut s = -steps;
            while s <= steps {
                let t = (s as f32) * voxel_size;
                let vx = ((px + ux * t) * inv_voxel).floor() as i32;
                let vy = ((py + uy * t) * inv_voxel).floor() as i32;
                let vz = ((pz + uz * t) * inv_voxel).floor() as i32;

                // Same occlusion cull as the update pass. Allocation must apply
                // exactly the same gate, or the pool fills with blocks that
                // never receive a contribution and the block count diverges
                // from the other arms while the meshes still match.
                let cx = ((vx as f32) + 0.5) * voxel_size;
                let cy = ((vy as f32) + 0.5) * voxel_size;
                let cz = ((vz as f32) + 0.5) * voxel_size;
                let (ex, ey, ez) = (cx - cam_x, cy - cam_y, cz - cam_z);
                if dist - (ex * ex + ey * ey + ez * ez).sqrt() >= -trunc {
                    find_or_insert::<true, false, true, true, true, false>(
                        table.as_ptr() as *mut i64,
                        hash_mask,
                        counters.as_ptr() as *mut i32,
                        pool_capacity,
                        drop_count.as_ptr() as *mut i64,
                        block_coord.as_ptr() as *mut i32,
                        floor_div(vx, BLOCK_DIM),
                        floor_div(vy, BLOCK_DIM),
                        floor_div(vz, BLOCK_DIM),
                    );
                }
                s += 1;
            }
        }
    }

    /// MEASUREMENT ONLY. The production configuration with every
    /// compare-exchange ATTEMPT tallied into `drop_count`. Correct apart from
    /// the overloaded counter. Measures the DYNAMIC CAS count, which static
    /// SASS cannot show: the two arms are at 1.02x static instruction parity
    /// (520 against 528) while A4 runs 5.9x slower.
    ///
    /// Exists to bisect the A4/A3 allocate gap; see docs/A3-A4-GAP.md.
    #[kernel]
    pub fn alloc_kernel_countcas(
        positions: &[f32],
        table: &[i64],
        counters: &[i32],
        block_coord: &[i32],
        drop_count: &[i64],
        n_points: i32,
        pool_capacity: i32,
        hash_mask: u32,
        voxel_size: f32,
        trunc: f32,
        cam_x: f32,
        cam_y: f32,
        cam_z: f32,
        radius_sq: f32,
    ) {
        let i = thread::index_1d().get() as i32;
        if i >= n_points {
            return;
        }
        unsafe {
            let p = positions.as_ptr().add((i * 3) as usize);
            let (px, py, pz) = (*p, *p.add(1), *p.add(2));
            let (dx, dy, dz) = (px - cam_x, py - cam_y, pz - cam_z);
            let d2 = dx * dx + dy * dy + dz * dz;
            if radius_sq > 0.0 && d2 > radius_sq {
                return;
            }
            let dist = d2.sqrt();
            if !(dist > 1e-6) {
                return;
            }
            let (ux, uy, uz) = (dx / dist, dy / dist, dz / dist);
            let inv_voxel = 1.0f32 / voxel_size;
            let steps = ((trunc * inv_voxel).ceil() as i32).max(1);

            let mut s = -steps;
            while s <= steps {
                let t = (s as f32) * voxel_size;
                let vx = ((px + ux * t) * inv_voxel).floor() as i32;
                let vy = ((py + uy * t) * inv_voxel).floor() as i32;
                let vz = ((pz + uz * t) * inv_voxel).floor() as i32;

                // Same occlusion cull as the update pass. Allocation must apply
                // exactly the same gate, or the pool fills with blocks that
                // never receive a contribution and the block count diverges
                // from the other arms while the meshes still match.
                let cx = ((vx as f32) + 0.5) * voxel_size;
                let cy = ((vy as f32) + 0.5) * voxel_size;
                let cz = ((vz as f32) + 0.5) * voxel_size;
                let (ex, ey, ez) = (cx - cam_x, cy - cam_y, cz - cam_z);
                if dist - (ex * ex + ey * ey + ez * ez).sqrt() >= -trunc {
                    find_or_insert::<true, true, true, true, true, true>(
                        table.as_ptr() as *mut i64,
                        hash_mask,
                        counters.as_ptr() as *mut i32,
                        pool_capacity,
                        drop_count.as_ptr() as *mut i64,
                        block_coord.as_ptr() as *mut i32,
                        floor_div(vx, BLOCK_DIM),
                        floor_div(vy, BLOCK_DIM),
                        floor_div(vz, BLOCK_DIM),
                    );
                }
                s += 1;
            }
        }
    }

    /// MEASUREMENT ONLY, and deliberately incorrect. Insert normally but never wait for a peer to publish `block_idx`. Returns a
    /// stale -1 for a block another thread is mid-inserting, so the block count
    /// is wrong. Prices the publication spin.
    ///
    /// Exists to bisect the A4/A3 allocate gap; see docs/A3-A4-GAP.md.
    #[kernel]
    pub fn alloc_kernel_nofence(
        positions: &[f32],
        table: &[i64],
        counters: &[i32],
        block_coord: &[i32],
        drop_count: &[i64],
        n_points: i32,
        pool_capacity: i32,
        hash_mask: u32,
        voxel_size: f32,
        trunc: f32,
        cam_x: f32,
        cam_y: f32,
        cam_z: f32,
        radius_sq: f32,
    ) {
        let i = thread::index_1d().get() as i32;
        if i >= n_points {
            return;
        }
        unsafe {
            let p = positions.as_ptr().add((i * 3) as usize);
            let (px, py, pz) = (*p, *p.add(1), *p.add(2));
            let (dx, dy, dz) = (px - cam_x, py - cam_y, pz - cam_z);
            let d2 = dx * dx + dy * dy + dz * dz;
            if radius_sq > 0.0 && d2 > radius_sq {
                return;
            }
            let dist = d2.sqrt();
            if !(dist > 1e-6) {
                return;
            }
            let (ux, uy, uz) = (dx / dist, dy / dist, dz / dist);
            let inv_voxel = 1.0f32 / voxel_size;
            let steps = ((trunc * inv_voxel).ceil() as i32).max(1);

            let mut s = -steps;
            while s <= steps {
                let t = (s as f32) * voxel_size;
                let vx = ((px + ux * t) * inv_voxel).floor() as i32;
                let vy = ((py + uy * t) * inv_voxel).floor() as i32;
                let vz = ((pz + uz * t) * inv_voxel).floor() as i32;

                // Same occlusion cull as the update pass. Allocation must apply
                // exactly the same gate, or the pool fills with blocks that
                // never receive a contribution and the block count diverges
                // from the other arms while the meshes still match.
                let cx = ((vx as f32) + 0.5) * voxel_size;
                let cy = ((vy as f32) + 0.5) * voxel_size;
                let cz = ((vz as f32) + 0.5) * voxel_size;
                let (ex, ey, ez) = (cx - cam_x, cy - cam_y, cz - cam_z);
                if dist - (ex * ex + ey * ey + ez * ez).sqrt() >= -trunc {
                    find_or_insert::<true, true, true, false, true, false>(
                        table.as_ptr() as *mut i64,
                        hash_mask,
                        counters.as_ptr() as *mut i32,
                        pool_capacity,
                        drop_count.as_ptr() as *mut i64,
                        block_coord.as_ptr() as *mut i32,
                        floor_div(vx, BLOCK_DIM),
                        floor_div(vy, BLOCK_DIM),
                        floor_div(vz, BLOCK_DIM),
                    );
                }
                s += 1;
            }
        }
    }

    /// MEASUREMENT ONLY, and deliberately incorrect. Insert normally but never wait for a peer to publish `block_idx`. Returns a
    /// stale -1 for a block another thread is mid-inserting, so the block count
    /// is wrong. Prices the publication spin.
    ///
    /// Exists to bisect the A4/A3 allocate gap; see docs/A3-A4-GAP.md.
    #[kernel]
    pub fn alloc_kernel_nopublish(
        positions: &[f32],
        table: &[i64],
        counters: &[i32],
        block_coord: &[i32],
        drop_count: &[i64],
        n_points: i32,
        pool_capacity: i32,
        hash_mask: u32,
        voxel_size: f32,
        trunc: f32,
        cam_x: f32,
        cam_y: f32,
        cam_z: f32,
        radius_sq: f32,
    ) {
        let i = thread::index_1d().get() as i32;
        if i >= n_points {
            return;
        }
        unsafe {
            let p = positions.as_ptr().add((i * 3) as usize);
            let (px, py, pz) = (*p, *p.add(1), *p.add(2));
            let (dx, dy, dz) = (px - cam_x, py - cam_y, pz - cam_z);
            let d2 = dx * dx + dy * dy + dz * dz;
            if radius_sq > 0.0 && d2 > radius_sq {
                return;
            }
            let dist = d2.sqrt();
            if !(dist > 1e-6) {
                return;
            }
            let (ux, uy, uz) = (dx / dist, dy / dist, dz / dist);
            let inv_voxel = 1.0f32 / voxel_size;
            let steps = ((trunc * inv_voxel).ceil() as i32).max(1);

            let mut s = -steps;
            while s <= steps {
                let t = (s as f32) * voxel_size;
                let vx = ((px + ux * t) * inv_voxel).floor() as i32;
                let vy = ((py + uy * t) * inv_voxel).floor() as i32;
                let vz = ((pz + uz * t) * inv_voxel).floor() as i32;

                // Same occlusion cull as the update pass. Allocation must apply
                // exactly the same gate, or the pool fills with blocks that
                // never receive a contribution and the block count diverges
                // from the other arms while the meshes still match.
                let cx = ((vx as f32) + 0.5) * voxel_size;
                let cy = ((vy as f32) + 0.5) * voxel_size;
                let cz = ((vz as f32) + 0.5) * voxel_size;
                let (ex, ey, ez) = (cx - cam_x, cy - cam_y, cz - cam_z);
                if dist - (ex * ex + ey * ey + ez * ez).sqrt() >= -trunc {
                    find_or_insert::<true, false, true, false, false, false>(
                        table.as_ptr() as *mut i64,
                        hash_mask,
                        counters.as_ptr() as *mut i32,
                        pool_capacity,
                        drop_count.as_ptr() as *mut i64,
                        block_coord.as_ptr() as *mut i32,
                        floor_div(vx, BLOCK_DIM),
                        floor_div(vy, BLOCK_DIM),
                        floor_div(vz, BLOCK_DIM),
                    );
                }
                s += 1;
            }
        }
    }

    /// MEASUREMENT ONLY, and deliberately incorrect. Probe and read, never compare-exchange, so nothing is ever inserted.
    /// Prices the CAS and everything downstream of a successful claim.
    ///
    /// Exists to bisect the A4/A3 allocate gap; see docs/A3-A4-GAP.md.
    #[kernel]
    pub fn alloc_kernel_nocas(
        positions: &[f32],
        table: &[i64],
        counters: &[i32],
        block_coord: &[i32],
        drop_count: &[i64],
        n_points: i32,
        pool_capacity: i32,
        hash_mask: u32,
        voxel_size: f32,
        trunc: f32,
        cam_x: f32,
        cam_y: f32,
        cam_z: f32,
        radius_sq: f32,
    ) {
        let i = thread::index_1d().get() as i32;
        if i >= n_points {
            return;
        }
        unsafe {
            let p = positions.as_ptr().add((i * 3) as usize);
            let (px, py, pz) = (*p, *p.add(1), *p.add(2));
            let (dx, dy, dz) = (px - cam_x, py - cam_y, pz - cam_z);
            let d2 = dx * dx + dy * dy + dz * dz;
            if radius_sq > 0.0 && d2 > radius_sq {
                return;
            }
            let dist = d2.sqrt();
            if !(dist > 1e-6) {
                return;
            }
            let (ux, uy, uz) = (dx / dist, dy / dist, dz / dist);
            let inv_voxel = 1.0f32 / voxel_size;
            let steps = ((trunc * inv_voxel).ceil() as i32).max(1);

            let mut s = -steps;
            while s <= steps {
                let t = (s as f32) * voxel_size;
                let vx = ((px + ux * t) * inv_voxel).floor() as i32;
                let vy = ((py + uy * t) * inv_voxel).floor() as i32;
                let vz = ((pz + uz * t) * inv_voxel).floor() as i32;

                // Same occlusion cull as the update pass. Allocation must apply
                // exactly the same gate, or the pool fills with blocks that
                // never receive a contribution and the block count diverges
                // from the other arms while the meshes still match.
                let cx = ((vx as f32) + 0.5) * voxel_size;
                let cy = ((vy as f32) + 0.5) * voxel_size;
                let cz = ((vz as f32) + 0.5) * voxel_size;
                let (ex, ey, ez) = (cx - cam_x, cy - cam_y, cz - cam_z);
                if dist - (ex * ex + ey * ey + ez * ez).sqrt() >= -trunc {
                    find_or_insert::<false, true, true, true, true, false>(
                        table.as_ptr() as *mut i64,
                        hash_mask,
                        counters.as_ptr() as *mut i32,
                        pool_capacity,
                        drop_count.as_ptr() as *mut i64,
                        block_coord.as_ptr() as *mut i32,
                        floor_div(vx, BLOCK_DIM),
                        floor_div(vy, BLOCK_DIM),
                        floor_div(vz, BLOCK_DIM),
                    );
                }
                s += 1;
            }
        }
    }

    /// MEASUREMENT ONLY, and deliberately incorrect. Insert with the CAS but derive the pool index from the block coordinate
    /// instead of the shared counter. Prices contention on that one address.
    ///
    /// Exists to bisect the A4/A3 allocate gap; see docs/A3-A4-GAP.md.
    #[kernel]
    pub fn alloc_kernel_nocount(
        positions: &[f32],
        table: &[i64],
        counters: &[i32],
        block_coord: &[i32],
        drop_count: &[i64],
        n_points: i32,
        pool_capacity: i32,
        hash_mask: u32,
        voxel_size: f32,
        trunc: f32,
        cam_x: f32,
        cam_y: f32,
        cam_z: f32,
        radius_sq: f32,
    ) {
        let i = thread::index_1d().get() as i32;
        if i >= n_points {
            return;
        }
        unsafe {
            let p = positions.as_ptr().add((i * 3) as usize);
            let (px, py, pz) = (*p, *p.add(1), *p.add(2));
            let (dx, dy, dz) = (px - cam_x, py - cam_y, pz - cam_z);
            let d2 = dx * dx + dy * dy + dz * dz;
            if radius_sq > 0.0 && d2 > radius_sq {
                return;
            }
            let dist = d2.sqrt();
            if !(dist > 1e-6) {
                return;
            }
            let (ux, uy, uz) = (dx / dist, dy / dist, dz / dist);
            let inv_voxel = 1.0f32 / voxel_size;
            let steps = ((trunc * inv_voxel).ceil() as i32).max(1);

            let mut s = -steps;
            while s <= steps {
                let t = (s as f32) * voxel_size;
                let vx = ((px + ux * t) * inv_voxel).floor() as i32;
                let vy = ((py + uy * t) * inv_voxel).floor() as i32;
                let vz = ((pz + uz * t) * inv_voxel).floor() as i32;

                // Same occlusion cull as the update pass. Allocation must apply
                // exactly the same gate, or the pool fills with blocks that
                // never receive a contribution and the block count diverges
                // from the other arms while the meshes still match.
                let cx = ((vx as f32) + 0.5) * voxel_size;
                let cy = ((vy as f32) + 0.5) * voxel_size;
                let cz = ((vz as f32) + 0.5) * voxel_size;
                let (ex, ey, ez) = (cx - cam_x, cy - cam_y, cz - cam_z);
                if dist - (ex * ex + ey * ey + ez * ez).sqrt() >= -trunc {
                    find_or_insert::<true, true, false, true, true, false>(
                        table.as_ptr() as *mut i64,
                        hash_mask,
                        counters.as_ptr() as *mut i32,
                        pool_capacity,
                        drop_count.as_ptr() as *mut i64,
                        block_coord.as_ptr() as *mut i32,
                        floor_div(vx, BLOCK_DIM),
                        floor_div(vy, BLOCK_DIM),
                        floor_div(vz, BLOCK_DIM),
                    );
                }
                s += 1;
            }
        }
    }

    /// Pass 2: accumulate weighted sums into already-allocated blocks.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn update_kernel(
        positions: &[f32],
        table: &[i64],
        tsdf: &[f32],
        weight: &[f32],
        r: &[f32],
        g: &[f32],
        b: &[f32],
        n_points: i32,
        hash_mask: u32,
        voxel_size: f32,
        trunc: f32,
        weight_cap: f32,
        cam_x: f32,
        cam_y: f32,
        cam_z: f32,
        radius_sq: f32,
    ) {
        let i = thread::index_1d().get() as i32;
        if i >= n_points {
            return;
        }
        unsafe {
            let p = positions.as_ptr().add((i * 3) as usize);
            let (px, py, pz) = (*p, *p.add(1), *p.add(2));
            let (dx, dy, dz) = (px - cam_x, py - cam_y, pz - cam_z);
            let d2 = dx * dx + dy * dy + dz * dz;
            if radius_sq > 0.0 && d2 > radius_sq {
                return;
            }
            let dist = d2.sqrt();
            if !(dist > 1e-6) {
                return;
            }
            let (ux, uy, uz) = (dx / dist, dy / dist, dz / dist);
            let inv_voxel = 1.0f32 / voxel_size;
            let inv_trunc = 1.0f32 / trunc;
            let steps = ((trunc * inv_voxel).ceil() as i32).max(1);

            let mut s = -steps;
            while s <= steps {
                let t = (s as f32) * voxel_size;
                let vx = ((px + ux * t) * inv_voxel).floor() as i32;
                let vy = ((py + uy * t) * inv_voxel).floor() as i32;
                let vz = ((pz + uz * t) * inv_voxel).floor() as i32;

                // Signed distance at the VOXEL CENTRE, not at the ray sample
                // that selected it. Points are binned with floor(p / voxel), so
                // the centre is up to half a voxel away; using the sample's own
                // offset biases every voxel and shows up as a uniform radius
                // error on curved surfaces (measured 0.005 m on a 0.01 m voxel).
                let cx = ((vx as f32) + 0.5) * voxel_size;
                let cy = ((vy as f32) + 0.5) * voxel_size;
                let cz = ((vz as f32) + 0.5) * voxel_size;
                let (ex, ey, ez) = (cx - cam_x, cy - cam_y, cz - cam_z);
                let sdf = dist - (ex * ex + ey * ey + ez * ez).sqrt();
                if sdf < -trunc {
                    s += 1;
                    continue;
                }

                let bx = floor_div(vx, BLOCK_DIM);
                let by = floor_div(vy, BLOCK_DIM);
                let bz = floor_div(vz, BLOCK_DIM);
                let bi = find_block(table.as_ptr() as *mut i64, hash_mask, bx, by, bz);
                if bi >= 0 {
                    let lx = vx - bx * BLOCK_DIM;
                    let ly = vy - by * BLOCK_DIM;
                    let lz = vz - bz * BLOCK_DIM;
                    let idx = (bi * BLOCK_VOXELS + (lz * BLOCK_DIM + ly) * BLOCK_DIM + lx) as usize;

                    let w_ptr = weight.as_ptr().add(idx) as *mut f32;
                    // Plain (non-atomic) read for the cap check, matching A3.
                    //
                    // Two reasons. The cap is explicitly approximate, so a
                    // torn or stale read only lets a few extra contributions
                    // land past the threshold, and enforcing it exactly would
                    // need the read-modify-write the weighted-sum form exists
                    // to avoid. And libNVVM rejects atomic float LOADS
                    // outright ("Atomic loads/stores are not supported"),
                    // which the default LLVM path accepts but
                    // --materialize-cubin does not, so an atomic load here
                    // would build in one mode and fail in the other.
                    let w_now = w_ptr.read_volatile();
                    if !(weight_cap > 0.0 && w_now >= weight_cap) {
                        let w_atomic = DeviceAtomicF32::from_ptr(w_ptr);
                        let sdf_n = (sdf * inv_trunc).clamp(-1.0, 1.0);
                        // Weighted SUMS, matching the other arms. Sums commute,
                        // so plain atomic adds suffice and no read-modify-write
                        // exists to race.
                        // All five accumulators, matching A3 exactly. Omitting
                        // colour would leave this arm doing 2 atomics per voxel
                        // against A3's 5, and the update comparison would be
                        // measuring a 40% workload difference rather than code
                        // generation. Colours are the neutral-grey default the
                        // other arms use when no colour buffer is supplied.
                        w_atomic.fetch_add(1.0, AtomicOrdering::Relaxed);
                        DeviceAtomicF32::from_ptr(tsdf.as_ptr().add(idx) as *mut f32)
                            .fetch_add(sdf_n, AtomicOrdering::Relaxed);
                        DeviceAtomicF32::from_ptr(r.as_ptr().add(idx) as *mut f32)
                            .fetch_add(128.0, AtomicOrdering::Relaxed);
                        DeviceAtomicF32::from_ptr(g.as_ptr().add(idx) as *mut f32)
                            .fetch_add(128.0, AtomicOrdering::Relaxed);
                        DeviceAtomicF32::from_ptr(b.as_ptr().add(idx) as *mut f32)
                            .fetch_add(128.0, AtomicOrdering::Relaxed);
                    }
                }
                s += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// C ABI: run A4's kernels against device state allocated by the C++ library.
//
// A4 allocates nothing. It receives the same hash table and voxel pool the CUDA
// C++ arm uses and runs its own kernels over them, which is what "arms differ
// in the integrate path, extraction is shared" requires in practice. It is also
// the shape arm A5a needs.
// ---------------------------------------------------------------------------

use cuda_core::DeviceBuffer;
use std::sync::{Arc, OnceLock};

/// Mirrors `OsnTsdfDeviceView` in include/osn_tsdf/c_api.h.
///
/// Field order and types must match EXACTLY. There is no checking across this
/// boundary: a missing or reordered field shifts everything after it, and the
/// symptom is a hang or garbage geometry rather than a link error. The C side
/// carries a static_assert on the struct size as a partial guard; keep the two
/// in step whenever either changes.
#[repr(C)]
pub struct DeviceViewC {
    pub table: u64,
    pub block_count: u64,
    pub drop_count: u64,
    pub block_coord: u64,
    pub tsdf: u64,
    pub weight: u64,
    pub r: u64,
    pub g: u64,
    pub b: u64,
    pub hash_mask: u32,
    /// First slot of the Triton scratch region. Unused by this arm, but the
    /// field must be present: this struct is a raw ABI mirror of
    /// `OsnTsdfDeviceView`, so an omitted field silently shifts every
    /// subsequent one. Omitting it made A4 read a garbage voxel size and spin
    /// forever, with no diagnostic.
    pub scratch_base: u32,
    pub pool_capacity: i32,
    pub block_dim: i32,
    pub voxel_size_m: f32,
    pub trunc_m: f32,
    pub weight_cap: f32,
}

/// Wrap a caller-owned device pointer as a `DeviceBuffer` for one launch,
/// then release it without freeing.
///
/// The buffers belong to the C++ volume. `DeviceBuffer` frees on drop, so every
/// borrow must be dismantled with `into_raw_parts`; forgetting one would leak
/// the Arc, and dropping one would free memory the C++ side still owns and hand
/// the next arm a dangling pool.
struct Borrowed<T> {
    buf: Option<DeviceBuffer<T>>,
}

impl<T> Borrowed<T> {
    /// # Safety
    /// `ptr` must be a live device allocation of at least `len` elements.
    unsafe fn new(ptr: u64, len: usize, ctx: Arc<cuda_core::CudaContext>) -> Self {
        Self { buf: Some(unsafe { DeviceBuffer::<T>::from_raw_parts(ptr, len, ctx) }) }
    }
    fn get(&self) -> &DeviceBuffer<T> {
        self.buf.as_ref().unwrap()
    }
}

impl<T> Drop for Borrowed<T> {
    fn drop(&mut self) {
        if let Some(b) = self.buf.take() {
            let _ = b.into_raw_parts();  // release ownership, do not free
        }
    }
}

struct Loaded {
    ctx: Arc<cuda_core::CudaContext>,
    module: kernels::LoadedModule,
}

// Loaded once and reused: reloading per call would put module-load time inside
// anything the harness measures.
static MODULE: OnceLock<Option<Loaded>> = OnceLock::new();

fn module() -> Option<&'static Loaded> {
    MODULE
        .get_or_init(|| {
            let ctx = CudaContext::new(0).ok()?;  // already an Arc
            let module = kernels::load(&ctx).ok()?;
            Some(Loaded { ctx, module })
        })
        .as_ref()
}

/// Runs the 128-bit compare-exchange self-test. `cell` is a 16-byte device
/// allocation preset to (-1, -1, 0); `out` receives four i64: the two words the
/// exchange returned, then the two words the cell holds afterwards.
///
/// # Safety
/// Both pointers must be live device allocations of the stated size.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn a4_cas128_selftest(cell: u64, out: u64) -> i32 {
    let Some(m) = module() else { return 1 };
    let c = unsafe { Borrowed::<i64>::new(cell, 2, m.ctx.clone()) };
    let o = unsafe { Borrowed::<i64>::new(out, 4, m.ctx.clone()) };
    let stream = m.ctx.default_stream();
    let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
    // SAFETY: shapes and extents match the kernel.
    let r = unsafe { m.module.cas128_selftest(&stream, cfg, c.get(), o.get()) };
    if r.is_ok() { 0 } else { 2 }
}

/// 0 on success. Loads the device module; safe to call repeatedly.
#[unsafe(no_mangle)]
pub extern "C" fn a4_init() -> i32 {
    if module().is_some() { 0 } else { 1 }
}

const THREADS: u32 = 256;

fn cfg_for(n: i32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((((n as u32) + THREADS - 1) / THREADS).max(1), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

const BLOCK_VOXELS_USZ: usize = (BLOCK_DIM * BLOCK_DIM * BLOCK_DIM) as usize;

/// # Safety
/// `view` must point to a valid `OsnTsdfDeviceView`, and `d_positions` must be
/// a device pointer covering `3 * n_points` floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn a4_allocate_blocks(
    view: *const DeviceViewC,
    d_positions: u64,
    n_points: i32,
    cam_x: f32,
    cam_y: f32,
    cam_z: f32,
    radius_m: f32,
) -> i32 {
    let Some(m) = module() else { return 1 };
    if view.is_null() || n_points <= 0 {
        return 2;
    }
    let v = unsafe { &*view };
    let n = n_points as usize;
    let pool = v.pool_capacity as usize;
    let table_len = ((v.hash_mask as usize) + 1) * 2; // 2 i64 words per slot

    let pos = unsafe { Borrowed::<f32>::new(d_positions, n * 3, m.ctx.clone()) };
    let table = unsafe { Borrowed::<i64>::new(v.table, table_len, m.ctx.clone()) };
    let counters = unsafe { Borrowed::<i32>::new(v.block_count, 1, m.ctx.clone()) };
    let coords = unsafe { Borrowed::<i32>::new(v.block_coord, pool * 3, m.ctx.clone()) };
    let drops = unsafe { Borrowed::<i64>::new(v.drop_count, 1, m.ctx.clone()) };

    let rsq = if radius_m > 0.0 { radius_m * radius_m } else { 0.0 };
    let stream = m.ctx.default_stream();
    // SAFETY: shapes and buffer extents match the kernel's accesses.
    let res = unsafe {
        m.module.alloc_kernel(
            &stream,
            cfg_for(n_points),
            pos.get(),
            table.get(),
            counters.get(),
            coords.get(),
            drops.get(),
            n_points,
            v.pool_capacity,
            v.hash_mask,
            v.voxel_size_m,
            v.trunc_m,
            cam_x,
            cam_y,
            cam_z,
            rsq,
        )
    };
    if res.is_ok() { 0 } else { 3 }
}

/// MEASUREMENT ONLY: launches a deliberately incorrect variant to price
/// one component of the allocate path. Never call this for real work.
///
/// # Safety
/// See `a4_allocate_blocks`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn a4_allocate_nospin(
    view: *const DeviceViewC,
    d_positions: u64,
    n_points: i32,
    cam_x: f32,
    cam_y: f32,
    cam_z: f32,
    radius_m: f32,
) -> i32 {
    let Some(m) = module() else { return 1 };
    if view.is_null() || n_points <= 0 {
        return 2;
    }
    let v = unsafe { &*view };
    let n = n_points as usize;
    let pool = v.pool_capacity as usize;
    let table_len = ((v.hash_mask as usize) + 1) * 2; // 2 i64 words per slot

    let pos = unsafe { Borrowed::<f32>::new(d_positions, n * 3, m.ctx.clone()) };
    let table = unsafe { Borrowed::<i64>::new(v.table, table_len, m.ctx.clone()) };
    let counters = unsafe { Borrowed::<i32>::new(v.block_count, 1, m.ctx.clone()) };
    let coords = unsafe { Borrowed::<i32>::new(v.block_coord, pool * 3, m.ctx.clone()) };
    let drops = unsafe { Borrowed::<i64>::new(v.drop_count, 1, m.ctx.clone()) };

    let rsq = if radius_m > 0.0 { radius_m * radius_m } else { 0.0 };
    let stream = m.ctx.default_stream();
    // SAFETY: shapes and buffer extents match the kernel's accesses.
    let res = unsafe {
        m.module.alloc_kernel_nospin(
            &stream,
            cfg_for(n_points),
            pos.get(),
            table.get(),
            counters.get(),
            coords.get(),
            drops.get(),
            n_points,
            v.pool_capacity,
            v.hash_mask,
            v.voxel_size_m,
            v.trunc_m,
            cam_x,
            cam_y,
            cam_z,
            rsq,
        )
    };
    if res.is_ok() { 0 } else { 3 }
}

/// Correct alternative implementation: the block index is the hash slot,
/// so there is no counter dependency, no publication and no wait.
///
/// # Safety
/// See `a4_allocate_blocks`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn a4_allocate_slotidx(
    view: *const DeviceViewC,
    d_positions: u64,
    n_points: i32,
    cam_x: f32,
    cam_y: f32,
    cam_z: f32,
    radius_m: f32,
) -> i32 {
    let Some(m) = module() else { return 1 };
    if view.is_null() || n_points <= 0 {
        return 2;
    }
    let v = unsafe { &*view };
    let n = n_points as usize;
    let pool = v.pool_capacity as usize;
    let table_len = ((v.hash_mask as usize) + 1) * 2; // 2 i64 words per slot

    let pos = unsafe { Borrowed::<f32>::new(d_positions, n * 3, m.ctx.clone()) };
    let table = unsafe { Borrowed::<i64>::new(v.table, table_len, m.ctx.clone()) };
    let counters = unsafe { Borrowed::<i32>::new(v.block_count, 1, m.ctx.clone()) };
    let coords = unsafe { Borrowed::<i32>::new(v.block_coord, pool * 3, m.ctx.clone()) };
    let drops = unsafe { Borrowed::<i64>::new(v.drop_count, 1, m.ctx.clone()) };

    let rsq = if radius_m > 0.0 { radius_m * radius_m } else { 0.0 };
    let stream = m.ctx.default_stream();
    // SAFETY: shapes and buffer extents match the kernel's accesses.
    let res = unsafe {
        m.module.alloc_kernel_slotidx(
            &stream,
            cfg_for(n_points),
            pos.get(),
            table.get(),
            counters.get(),
            coords.get(),
            drops.get(),
            n_points,
            v.pool_capacity,
            v.hash_mask,
            v.voxel_size_m,
            v.trunc_m,
            cam_x,
            cam_y,
            cam_z,
            rsq,
        )
    };
    if res.is_ok() { 0 } else { 3 }
}

/// Correct alternative implementation: publishes key and index in one
/// 128-bit exchange, so no reader waits. See `find_or_insert_128`.
///
/// # Safety
/// See `a4_allocate_blocks`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn a4_allocate_cas128(
    view: *const DeviceViewC,
    d_positions: u64,
    n_points: i32,
    cam_x: f32,
    cam_y: f32,
    cam_z: f32,
    radius_m: f32,
) -> i32 {
    let Some(m) = module() else { return 1 };
    if view.is_null() || n_points <= 0 {
        return 2;
    }
    let v = unsafe { &*view };
    let n = n_points as usize;
    let pool = v.pool_capacity as usize;
    let table_len = ((v.hash_mask as usize) + 1) * 2; // 2 i64 words per slot

    let pos = unsafe { Borrowed::<f32>::new(d_positions, n * 3, m.ctx.clone()) };
    let table = unsafe { Borrowed::<i64>::new(v.table, table_len, m.ctx.clone()) };
    let counters = unsafe { Borrowed::<i32>::new(v.block_count, 1, m.ctx.clone()) };
    let coords = unsafe { Borrowed::<i32>::new(v.block_coord, pool * 3, m.ctx.clone()) };
    let drops = unsafe { Borrowed::<i64>::new(v.drop_count, 1, m.ctx.clone()) };

    let rsq = if radius_m > 0.0 { radius_m * radius_m } else { 0.0 };
    let stream = m.ctx.default_stream();
    // SAFETY: shapes and buffer extents match the kernel's accesses.
    let res = unsafe {
        m.module.alloc_kernel_cas128(
            &stream,
            cfg_for(n_points),
            pos.get(),
            table.get(),
            counters.get(),
            coords.get(),
            drops.get(),
            n_points,
            v.pool_capacity,
            v.hash_mask,
            v.voxel_size_m,
            v.trunc_m,
            cam_x,
            cam_y,
            cam_z,
            rsq,
        )
    };
    if res.is_ok() { 0 } else { 3 }
}

/// MEASUREMENT ONLY: launches a deliberately incorrect variant to price
/// one component of the allocate path. Never call this for real work.
///
/// # Safety
/// See `a4_allocate_blocks`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn a4_allocate_countcas(
    view: *const DeviceViewC,
    d_positions: u64,
    n_points: i32,
    cam_x: f32,
    cam_y: f32,
    cam_z: f32,
    radius_m: f32,
) -> i32 {
    let Some(m) = module() else { return 1 };
    if view.is_null() || n_points <= 0 {
        return 2;
    }
    let v = unsafe { &*view };
    let n = n_points as usize;
    let pool = v.pool_capacity as usize;
    let table_len = ((v.hash_mask as usize) + 1) * 2; // 2 i64 words per slot

    let pos = unsafe { Borrowed::<f32>::new(d_positions, n * 3, m.ctx.clone()) };
    let table = unsafe { Borrowed::<i64>::new(v.table, table_len, m.ctx.clone()) };
    let counters = unsafe { Borrowed::<i32>::new(v.block_count, 1, m.ctx.clone()) };
    let coords = unsafe { Borrowed::<i32>::new(v.block_coord, pool * 3, m.ctx.clone()) };
    let drops = unsafe { Borrowed::<i64>::new(v.drop_count, 1, m.ctx.clone()) };

    let rsq = if radius_m > 0.0 { radius_m * radius_m } else { 0.0 };
    let stream = m.ctx.default_stream();
    // SAFETY: shapes and buffer extents match the kernel's accesses.
    let res = unsafe {
        m.module.alloc_kernel_countcas(
            &stream,
            cfg_for(n_points),
            pos.get(),
            table.get(),
            counters.get(),
            coords.get(),
            drops.get(),
            n_points,
            v.pool_capacity,
            v.hash_mask,
            v.voxel_size_m,
            v.trunc_m,
            cam_x,
            cam_y,
            cam_z,
            rsq,
        )
    };
    if res.is_ok() { 0 } else { 3 }
}

/// MEASUREMENT ONLY: launches a deliberately incorrect variant to price
/// one component of the allocate path. Never call this for real work.
///
/// # Safety
/// See `a4_allocate_blocks`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn a4_allocate_nopublish(
    view: *const DeviceViewC,
    d_positions: u64,
    n_points: i32,
    cam_x: f32,
    cam_y: f32,
    cam_z: f32,
    radius_m: f32,
) -> i32 {
    let Some(m) = module() else { return 1 };
    if view.is_null() || n_points <= 0 {
        return 2;
    }
    let v = unsafe { &*view };
    let n = n_points as usize;
    let pool = v.pool_capacity as usize;
    let table_len = ((v.hash_mask as usize) + 1) * 2; // 2 i64 words per slot

    let pos = unsafe { Borrowed::<f32>::new(d_positions, n * 3, m.ctx.clone()) };
    let table = unsafe { Borrowed::<i64>::new(v.table, table_len, m.ctx.clone()) };
    let counters = unsafe { Borrowed::<i32>::new(v.block_count, 1, m.ctx.clone()) };
    let coords = unsafe { Borrowed::<i32>::new(v.block_coord, pool * 3, m.ctx.clone()) };
    let drops = unsafe { Borrowed::<i64>::new(v.drop_count, 1, m.ctx.clone()) };

    let rsq = if radius_m > 0.0 { radius_m * radius_m } else { 0.0 };
    let stream = m.ctx.default_stream();
    // SAFETY: shapes and buffer extents match the kernel's accesses.
    let res = unsafe {
        m.module.alloc_kernel_nopublish(
            &stream,
            cfg_for(n_points),
            pos.get(),
            table.get(),
            counters.get(),
            coords.get(),
            drops.get(),
            n_points,
            v.pool_capacity,
            v.hash_mask,
            v.voxel_size_m,
            v.trunc_m,
            cam_x,
            cam_y,
            cam_z,
            rsq,
        )
    };
    if res.is_ok() { 0 } else { 3 }
}

/// MEASUREMENT ONLY: launches a deliberately incorrect variant to price
/// one component of the allocate path. Never call this for real work.
///
/// # Safety
/// See `a4_allocate_blocks`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn a4_allocate_nofence(
    view: *const DeviceViewC,
    d_positions: u64,
    n_points: i32,
    cam_x: f32,
    cam_y: f32,
    cam_z: f32,
    radius_m: f32,
) -> i32 {
    let Some(m) = module() else { return 1 };
    if view.is_null() || n_points <= 0 {
        return 2;
    }
    let v = unsafe { &*view };
    let n = n_points as usize;
    let pool = v.pool_capacity as usize;
    let table_len = ((v.hash_mask as usize) + 1) * 2; // 2 i64 words per slot

    let pos = unsafe { Borrowed::<f32>::new(d_positions, n * 3, m.ctx.clone()) };
    let table = unsafe { Borrowed::<i64>::new(v.table, table_len, m.ctx.clone()) };
    let counters = unsafe { Borrowed::<i32>::new(v.block_count, 1, m.ctx.clone()) };
    let coords = unsafe { Borrowed::<i32>::new(v.block_coord, pool * 3, m.ctx.clone()) };
    let drops = unsafe { Borrowed::<i64>::new(v.drop_count, 1, m.ctx.clone()) };

    let rsq = if radius_m > 0.0 { radius_m * radius_m } else { 0.0 };
    let stream = m.ctx.default_stream();
    // SAFETY: shapes and buffer extents match the kernel's accesses.
    let res = unsafe {
        m.module.alloc_kernel_nofence(
            &stream,
            cfg_for(n_points),
            pos.get(),
            table.get(),
            counters.get(),
            coords.get(),
            drops.get(),
            n_points,
            v.pool_capacity,
            v.hash_mask,
            v.voxel_size_m,
            v.trunc_m,
            cam_x,
            cam_y,
            cam_z,
            rsq,
        )
    };
    if res.is_ok() { 0 } else { 3 }
}

/// MEASUREMENT ONLY: launches a deliberately incorrect variant to price
/// one component of the allocate path. Never call this for real work.
///
/// # Safety
/// See `a4_allocate_blocks`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn a4_allocate_nocas(
    view: *const DeviceViewC,
    d_positions: u64,
    n_points: i32,
    cam_x: f32,
    cam_y: f32,
    cam_z: f32,
    radius_m: f32,
) -> i32 {
    let Some(m) = module() else { return 1 };
    if view.is_null() || n_points <= 0 {
        return 2;
    }
    let v = unsafe { &*view };
    let n = n_points as usize;
    let pool = v.pool_capacity as usize;
    let table_len = ((v.hash_mask as usize) + 1) * 2; // 2 i64 words per slot

    let pos = unsafe { Borrowed::<f32>::new(d_positions, n * 3, m.ctx.clone()) };
    let table = unsafe { Borrowed::<i64>::new(v.table, table_len, m.ctx.clone()) };
    let counters = unsafe { Borrowed::<i32>::new(v.block_count, 1, m.ctx.clone()) };
    let coords = unsafe { Borrowed::<i32>::new(v.block_coord, pool * 3, m.ctx.clone()) };
    let drops = unsafe { Borrowed::<i64>::new(v.drop_count, 1, m.ctx.clone()) };

    let rsq = if radius_m > 0.0 { radius_m * radius_m } else { 0.0 };
    let stream = m.ctx.default_stream();
    // SAFETY: shapes and buffer extents match the kernel's accesses.
    let res = unsafe {
        m.module.alloc_kernel_nocas(
            &stream,
            cfg_for(n_points),
            pos.get(),
            table.get(),
            counters.get(),
            coords.get(),
            drops.get(),
            n_points,
            v.pool_capacity,
            v.hash_mask,
            v.voxel_size_m,
            v.trunc_m,
            cam_x,
            cam_y,
            cam_z,
            rsq,
        )
    };
    if res.is_ok() { 0 } else { 3 }
}

/// MEASUREMENT ONLY: launches a deliberately incorrect variant to price
/// one component of the allocate path. Never call this for real work.
///
/// # Safety
/// See `a4_allocate_blocks`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn a4_allocate_nocount(
    view: *const DeviceViewC,
    d_positions: u64,
    n_points: i32,
    cam_x: f32,
    cam_y: f32,
    cam_z: f32,
    radius_m: f32,
) -> i32 {
    let Some(m) = module() else { return 1 };
    if view.is_null() || n_points <= 0 {
        return 2;
    }
    let v = unsafe { &*view };
    let n = n_points as usize;
    let pool = v.pool_capacity as usize;
    let table_len = ((v.hash_mask as usize) + 1) * 2; // 2 i64 words per slot

    let pos = unsafe { Borrowed::<f32>::new(d_positions, n * 3, m.ctx.clone()) };
    let table = unsafe { Borrowed::<i64>::new(v.table, table_len, m.ctx.clone()) };
    let counters = unsafe { Borrowed::<i32>::new(v.block_count, 1, m.ctx.clone()) };
    let coords = unsafe { Borrowed::<i32>::new(v.block_coord, pool * 3, m.ctx.clone()) };
    let drops = unsafe { Borrowed::<i64>::new(v.drop_count, 1, m.ctx.clone()) };

    let rsq = if radius_m > 0.0 { radius_m * radius_m } else { 0.0 };
    let stream = m.ctx.default_stream();
    // SAFETY: shapes and buffer extents match the kernel's accesses.
    let res = unsafe {
        m.module.alloc_kernel_nocount(
            &stream,
            cfg_for(n_points),
            pos.get(),
            table.get(),
            counters.get(),
            coords.get(),
            drops.get(),
            n_points,
            v.pool_capacity,
            v.hash_mask,
            v.voxel_size_m,
            v.trunc_m,
            cam_x,
            cam_y,
            cam_z,
            rsq,
        )
    };
    if res.is_ok() { 0 } else { 3 }
}

/// # Safety
/// See `a4_allocate_blocks`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn a4_update_voxels(
    view: *const DeviceViewC,
    d_positions: u64,
    n_points: i32,
    cam_x: f32,
    cam_y: f32,
    cam_z: f32,
    radius_m: f32,
) -> i32 {
    let Some(m) = module() else { return 1 };
    if view.is_null() || n_points <= 0 {
        return 2;
    }
    let v = unsafe { &*view };
    let n = n_points as usize;
    let n_vox = (v.pool_capacity as usize) * BLOCK_VOXELS_USZ;
    let table_len = ((v.hash_mask as usize) + 1) * 2;

    let pos = unsafe { Borrowed::<f32>::new(d_positions, n * 3, m.ctx.clone()) };
    let table = unsafe { Borrowed::<i64>::new(v.table, table_len, m.ctx.clone()) };
    let tsdf = unsafe { Borrowed::<f32>::new(v.tsdf, n_vox, m.ctx.clone()) };
    let weight = unsafe { Borrowed::<f32>::new(v.weight, n_vox, m.ctx.clone()) };
    let cr = unsafe { Borrowed::<f32>::new(v.r, n_vox, m.ctx.clone()) };
    let cg = unsafe { Borrowed::<f32>::new(v.g, n_vox, m.ctx.clone()) };
    let cb = unsafe { Borrowed::<f32>::new(v.b, n_vox, m.ctx.clone()) };

    let rsq = if radius_m > 0.0 { radius_m * radius_m } else { 0.0 };
    let stream = m.ctx.default_stream();
    // SAFETY: shapes and buffer extents match the kernel's accesses.
    let res = unsafe {
        m.module.update_kernel(
            &stream,
            cfg_for(n_points),
            pos.get(),
            table.get(),
            tsdf.get(),
            weight.get(),
            cr.get(),
            cg.get(),
            cb.get(),
            n_points,
            v.hash_mask,
            v.voxel_size_m,
            v.trunc_m,
            v.weight_cap,
            cam_x,
            cam_y,
            cam_z,
            rsq,
        )
    };
    if res.is_ok() { 0 } else { 3 }
}

/// Block until A4's queued work finishes. Required before timing or reading.
#[unsafe(no_mangle)]
pub extern "C" fn a4_synchronize() -> i32 {
    let Some(m) = module() else { return 1 };
    if m.ctx.default_stream().synchronize().is_ok() { 0 } else { 2 }
}
