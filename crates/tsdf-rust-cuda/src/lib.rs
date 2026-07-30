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
                // Plain volatile read, matching A3. libNVVM supports atomic RMW
            // but rejects atomic loads/stores outright, and A3 reads the key
            // plainly too, so this keeps the arms identical rather than giving
            // one of them stronger ordering than the other.
            let key = key_ptr.read_volatile();
                if key == EMPTY_KEY {
                    return -1;
                }
                if key == want {
                    // Spin until the winner publishes the index; seeing the key
                    // without the index would read -1 and drop the point.
                    let mut idx = idx_ptr.read_volatile();
                    while idx < 0 {
                        idx = idx_ptr.read_volatile();
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
                let key = key_ptr.read_volatile();

                if key == want {
                    let mut idx = idx_ptr.read_volatile();
                    while idx < 0 {
                        idx = idx_ptr.read_volatile();
                    }
                    return idx;
                }
                if key == EMPTY_KEY {
                    match key_atomic.compare_exchange(
                        EMPTY_KEY,
                        want,
                        AtomicOrdering::Relaxed,
                    AtomicOrdering::Relaxed,
                    ) {
                        Ok(_) => {
                            let idx = DeviceAtomicI32::from_ptr(counters)
                                .fetch_add(1, AtomicOrdering::Relaxed);
                            if idx >= pool_capacity {
                                DeviceAtomicI32::from_ptr(counters)
                                    .fetch_sub(1, AtomicOrdering::Relaxed);
                                // Atomic exchange, not a store: libNVVM allows RMW
                            // but not atomic stores. Relaxed, because a Release
                            // ordering lowers to an LLVM `fence`, which libNVVM
                            // also rejects ("Illegal instruction: fence"). The
                            // ordering comes from an explicit threadfence, the
                            // same construct A3 uses.
                            threadfence();
                            key_atomic.swap(EMPTY_KEY, AtomicOrdering::Relaxed);
                                DeviceAtomicI64::from_ptr(drop_count)
                                    .fetch_add(1, AtomicOrdering::Relaxed);
                                return -1;
                            }
                            let bc = block_coord.add((idx * 3) as usize);
                            bc.write(bx);
                            bc.add(1).write(by);
                            bc.add(2).write(bz);
                            // Publish the index only after the coordinate writes
                        // are visible. Explicit threadfence + relaxed exchange,
                        // exactly what A3 does (__threadfence + atomicExch):
                        // a Release-ordered RMW would lower to an LLVM fence,
                        // which libNVVM rejects.
                        threadfence();
                        DeviceAtomicI32::from_ptr(idx_ptr).swap(idx, AtomicOrdering::Relaxed);
                            return idx;
                        }
                        Err(actual) => {
                            if actual == want {
                                let mut idx =
                                    idx_ptr.read_volatile();
                                while idx < 0 {
                                    idx = idx_ptr.read_volatile();
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
                    find_or_insert(
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

/// Mirrors `OsnTsdfDeviceView` in include/osn_tsdf/c_api.h. Field order and
/// types must match exactly; this is a raw ABI boundary with no checking.
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
