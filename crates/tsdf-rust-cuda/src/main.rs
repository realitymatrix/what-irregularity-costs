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

use cuda_core::CudaContext;
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32, DeviceAtomicI32, DeviceAtomicI64};
use cuda_device::{kernel, thread};
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
                let key = DeviceAtomicI64::from_ptr(key_ptr).load(AtomicOrdering::Acquire);
                if key == EMPTY_KEY {
                    return -1;
                }
                if key == want {
                    // Spin until the winner publishes the index; seeing the key
                    // without the index would read -1 and drop the point.
                    let mut idx = DeviceAtomicI32::from_ptr(idx_ptr).load(AtomicOrdering::Acquire);
                    while idx < 0 {
                        idx = DeviceAtomicI32::from_ptr(idx_ptr).load(AtomicOrdering::Acquire);
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
    unsafe fn find_or_insert(
        table: *mut i64,
        mask: u32,
        counters: *mut i32,
        pool_capacity: i32,
        drop_count: *mut i32,
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
                let key = key_atomic.load(AtomicOrdering::Acquire);

                if key == want {
                    let mut idx = DeviceAtomicI32::from_ptr(idx_ptr).load(AtomicOrdering::Acquire);
                    while idx < 0 {
                        idx = DeviceAtomicI32::from_ptr(idx_ptr).load(AtomicOrdering::Acquire);
                    }
                    return idx;
                }
                if key == EMPTY_KEY {
                    match key_atomic.compare_exchange(
                        EMPTY_KEY,
                        want,
                        AtomicOrdering::AcqRel,
                        AtomicOrdering::Acquire,
                    ) {
                        Ok(_) => {
                            let idx = DeviceAtomicI32::from_ptr(counters)
                                .fetch_add(1, AtomicOrdering::AcqRel);
                            if idx >= pool_capacity {
                                DeviceAtomicI32::from_ptr(counters)
                                    .fetch_sub(1, AtomicOrdering::AcqRel);
                                key_atomic.store(EMPTY_KEY, AtomicOrdering::Release);
                                DeviceAtomicI32::from_ptr(drop_count)
                                    .fetch_add(1, AtomicOrdering::Relaxed);
                                return -1;
                            }
                            let bc = block_coord.add((idx * 3) as usize);
                            bc.write(bx);
                            bc.add(1).write(by);
                            bc.add(2).write(bz);
                            DeviceAtomicI32::from_ptr(idx_ptr).store(idx, AtomicOrdering::Release);
                            return idx;
                        }
                        Err(actual) => {
                            if actual == want {
                                let mut idx =
                                    DeviceAtomicI32::from_ptr(idx_ptr).load(AtomicOrdering::Acquire);
                                while idx < 0 {
                                    idx = DeviceAtomicI32::from_ptr(idx_ptr)
                                        .load(AtomicOrdering::Acquire);
                                }
                                return idx;
                            }
                        }
                    }
                }
                probe += 1;
            }
            DeviceAtomicI32::from_ptr(drop_count).fetch_add(1, AtomicOrdering::Relaxed);
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
        drop_count: &[i32],
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
                    find_or_insert(
                        table.as_ptr() as *mut i64,
                        hash_mask,
                        counters.as_ptr() as *mut i32,
                        pool_capacity,
                        drop_count.as_ptr() as *mut i32,
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
                    let w_atomic = DeviceAtomicF32::from_ptr(w_ptr);
                    // Approximate cap, read without ordering: enforcing it
                    // exactly would need a read-modify-write, which is what the
                    // weighted-sum form exists to avoid.
                    if !(weight_cap > 0.0 && w_atomic.load(AtomicOrdering::Relaxed) >= weight_cap) {
                        let sdf_n = (sdf * inv_trunc).clamp(-1.0, 1.0);
                        // Weighted SUMS, matching the other arms. Sums commute,
                        // so plain atomic adds suffice and no read-modify-write
                        // exists to race.
                        w_atomic.fetch_add(1.0, AtomicOrdering::Relaxed);
                        DeviceAtomicF32::from_ptr(tsdf.as_ptr().add(idx) as *mut f32)
                            .fetch_add(sdf_n, AtomicOrdering::Relaxed);
                    }
                }
                s += 1;
            }
        }
    }
}

/// Smoke entry point. The arm is driven by the project harness, which loads the
/// materialised cubin, so this only confirms the module builds and a context
/// can be created on the target device.
fn main() {
    println!("=== A4 (Rust CUDA via cuda-oxide) ===");
    match CudaContext::new(0) {
        Ok(_) => println!("  context ok; kernels: alloc_kernel, update_kernel"),
        Err(e) => {
            eprintln!("  no CUDA device: {e}");
            std::process::exit(1);
        }
    }
}
