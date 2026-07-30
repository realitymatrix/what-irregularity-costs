//! Spike S1 for the tsdf-backends project.
//!
//! Question: can cuda-oxide express the two primitives the hash TSDF integrate
//! path actually needs, and do they run correctly on sm_120 (RTX 50 series)?
//!
//! Mirrors the real kernel in
//! `openstrate-reconstruct-rs/vendor/libinfer/src/tsdf_hash.cu`:
//!   - `hash_integrate_kernel`  : atomicAdd accumulation into an SoA voxel pool
//!   - line 338                 : atomicCAS linear-probe hash insertion
//!
//! Test 1 (insert): many threads race to insert block coords into an
//! open-addressed table with linear probing. Exactly one winner per distinct
//! coord; losers must find the existing slot. Verifies the CAS-claim protocol,
//! not just that CAS compiles.
//!
//! Test 2 (accumulate): threads atomicAdd weighted position sums + weight +
//! count into the claimed slot, the same per-voxel accumulator layout the real
//! code uses. Verifies f32 and u32 device-scope atomics under contention.
//!
//! Build:
//!   cargo oxide run tsdf_hash_spike --arch sm_120

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32, DeviceAtomicU32};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

/// Table slots. Power of two so the mask-based hash matches the CUDA original.
const HASH_SIZE: usize = 1024;
/// Sentinel for an empty slot. The real code uses a dedicated HASH_EMPTY_KEY.
const HASH_EMPTY_KEY: u32 = 0xFFFF_FFFF;
/// Distinct block coords contended for.
const N_COORDS: u32 = 64;
/// Threads per coord, i.e. the contention factor.
const DUP: u32 = 8;
const THREADS: usize = (N_COORDS * DUP) as usize;
/// Max linear probes before giving up, as in the CUDA version.
const MAX_PROBE: u32 = 64;

#[cuda_module]
mod kernels {
    use super::*;

    /// atomicCAS linear-probe insert, mirroring tsdf_hash.cu:338.
    ///
    /// `table` holds the key per slot (HASH_EMPTY_KEY when free).
    /// `slot_out` receives the slot each thread resolved to.
    /// `won_out` is 1 for the thread that claimed the slot, 0 for a thread
    /// that found it already claimed.
    #[kernel]
    pub fn hash_insert(
        table: &[u32],
        mut slot_out: DisjointSlice<u32>,
        mut won_out: DisjointSlice<u32>,
    ) {
        let gid = thread::index_1d();
        let tid = gid.get() as u32;
        if tid >= THREADS as u32 {
            return;
        }

        // Many threads per coord, so every coord is genuinely contended.
        let key = tid / DUP;

        // Same cheap integer mix the CUDA side uses, then mask to table size.
        let mut h = key.wrapping_mul(73_856_093) ^ key.wrapping_mul(19_349_663);
        h &= (HASH_SIZE as u32) - 1;

        let mut resolved: u32 = HASH_EMPTY_KEY;
        let mut won: u32 = 0;

        let mut probe = 0u32;
        while probe < MAX_PROBE {
            let idx = ((h + probe) & ((HASH_SIZE as u32) - 1)) as usize;

            // SAFETY: idx is masked into [0, HASH_SIZE) and the host allocates
            // HASH_SIZE u32 slots. Shared mutation goes only through the atomic.
            let cell = unsafe { &*(table.as_ptr().add(idx) as *const DeviceAtomicU32) };

            let observed = cell.load(AtomicOrdering::Acquire);
            if observed == key {
                // Already inserted by a peer: this is our slot.
                resolved = idx as u32;
                break;
            }
            if observed == HASH_EMPTY_KEY {
                match cell.compare_exchange(
                    HASH_EMPTY_KEY,
                    key,
                    AtomicOrdering::AcqRel,
                    AtomicOrdering::Relaxed,
                ) {
                    Ok(_) => {
                        resolved = idx as u32;
                        won = 1;
                        break;
                    }
                    Err(actual) => {
                        // Lost the race. If the winner wrote our key, share it.
                        if actual == key {
                            resolved = idx as u32;
                            break;
                        }
                        // Otherwise a different key took the slot: keep probing.
                    }
                }
            }
            probe += 1;
        }

        // ThreadIndex is not Copy, so take a fresh one for the second slice.
        if let Some(s) = slot_out.get_mut(gid) {
            *s = resolved;
        }
        if let Some(w) = won_out.get_mut(thread::index_1d()) {
            *w = won;
        }
    }

    /// Per-voxel accumulation, mirroring hash_integrate_kernel's SoA update.
    /// Each thread adds (1,2,3) to the position sums, 0.5 to weight, 1 to count.
    #[kernel]
    pub fn voxel_accumulate(
        slot_in: &[u32],
        sum_xyz: &[f32],
        weight: &[f32],
        count: &[u32],
    ) {
        let gid = thread::index_1d();
        let tid = gid.get();
        if tid >= THREADS {
            return;
        }

        let slot = slot_in[tid];
        if slot == HASH_EMPTY_KEY {
            return;
        }
        let s = slot as usize;

        // SAFETY: slot < HASH_SIZE by construction in hash_insert; host
        // allocates HASH_SIZE*3 floats for sum_xyz and HASH_SIZE for the rest.
        unsafe {
            let sx = &*(sum_xyz.as_ptr().add(s * 3) as *const DeviceAtomicF32);
            let sy = &*(sum_xyz.as_ptr().add(s * 3 + 1) as *const DeviceAtomicF32);
            let sz = &*(sum_xyz.as_ptr().add(s * 3 + 2) as *const DeviceAtomicF32);
            sx.fetch_add(1.0, AtomicOrdering::Relaxed);
            sy.fetch_add(2.0, AtomicOrdering::Relaxed);
            sz.fetch_add(3.0, AtomicOrdering::Relaxed);

            let w = &*(weight.as_ptr().add(s) as *const DeviceAtomicF32);
            w.fetch_add(0.5, AtomicOrdering::Relaxed);

            let c = &*(count.as_ptr().add(s) as *const DeviceAtomicU32);
            c.fetch_add(1, AtomicOrdering::Relaxed);
        }
    }
}

fn main() {
    println!("=== TSDF hash-insert + accumulate spike (cuda-oxide) ===\n");

    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.default_stream();

    let module = ctx
        .load_module_from_file("tsdf_hash_spike.ptx")
        .expect("failed to load PTX module");
    let module = kernels::from_module(module).expect("failed to init typed module");

    let cfg = LaunchConfig {
        grid_dim: (((THREADS + 255) / 256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut pass = true;

    // -- Test 1: atomicCAS linear-probe insertion ---------------------------
    println!("--- Test 1: atomicCAS linear-probe hash insert ---");
    println!("  {THREADS} threads racing for {N_COORDS} distinct coords ({DUP}x contention)");
    let table = vec![HASH_EMPTY_KEY; HASH_SIZE];
    let table_dev = DeviceBuffer::from_host(&stream, &table).unwrap();
    let mut slot_dev = DeviceBuffer::<u32>::zeroed(&stream, THREADS).unwrap();
    let mut won_dev = DeviceBuffer::<u32>::zeroed(&stream, THREADS).unwrap();

    // SAFETY: launch shape matches; buffers cover all kernel accesses.
    unsafe {
        module.hash_insert(
            (stream).as_ref(),
            cfg,
            &table_dev,
            &mut slot_dev,
            &mut won_dev,
        )
    }
    .expect("hash_insert launch failed");
    stream.synchronize().unwrap();

    let slots = slot_dev.to_host_vec(&stream).unwrap();
    let wons = won_dev.to_host_vec(&stream).unwrap();

    let winners: u32 = wons.iter().sum();
    let unresolved = slots.iter().filter(|&&s| s == HASH_EMPTY_KEY).count();
    println!("  winners = {winners} (expected {N_COORDS})");
    println!("  unresolved threads = {unresolved} (expected 0)");

    // Every thread sharing a coord must agree on the slot.
    let mut agree = true;
    for c in 0..N_COORDS {
        let base = slots[(c * DUP) as usize];
        for d in 0..DUP {
            if slots[(c * DUP + d) as usize] != base {
                agree = false;
            }
        }
    }
    println!("  all duplicates agree on slot = {agree}");
    // Distinct coords must occupy distinct slots.
    let mut occupied: Vec<u32> = (0..N_COORDS).map(|c| slots[(c * DUP) as usize]).collect();
    occupied.sort_unstable();
    occupied.dedup();
    let distinct = occupied.len();
    println!("  distinct slots used = {distinct} (expected {N_COORDS})");

    let t1 = winners == N_COORDS && unresolved == 0 && agree && distinct == N_COORDS as usize;
    println!("  => Test 1 {}\n", if t1 { "PASS" } else { "FAIL" });
    pass &= t1;

    // -- Test 2: SoA atomicAdd accumulation ---------------------------------
    println!("--- Test 2: atomicAdd SoA voxel accumulation ---");
    let sum_dev = DeviceBuffer::<f32>::zeroed(&stream, HASH_SIZE * 3).unwrap();
    let w_dev = DeviceBuffer::<f32>::zeroed(&stream, HASH_SIZE).unwrap();
    let c_dev = DeviceBuffer::<u32>::zeroed(&stream, HASH_SIZE).unwrap();

    // SAFETY: launch shape matches; buffers cover all kernel accesses.
    unsafe {
        module.voxel_accumulate((stream).as_ref(), cfg, &slot_dev, &sum_dev, &w_dev, &c_dev)
    }
    .expect("voxel_accumulate launch failed");
    stream.synchronize().unwrap();

    let sums = sum_dev.to_host_vec(&stream).unwrap();
    let ws = w_dev.to_host_vec(&stream).unwrap();
    let cs = c_dev.to_host_vec(&stream).unwrap();

    let total_count: u32 = cs.iter().sum();
    let total_w: f32 = ws.iter().sum();
    let total_x: f32 = (0..HASH_SIZE).map(|i| sums[i * 3]).sum();
    let total_z: f32 = (0..HASH_SIZE).map(|i| sums[i * 3 + 2]).sum();

    let exp_count = THREADS as u32;
    let exp_w = THREADS as f32 * 0.5;
    println!("  total count  = {total_count} (expected {exp_count})");
    println!("  total weight = {total_w} (expected {exp_w})");
    println!("  total sum_x  = {total_x} (expected {})", THREADS as f32);
    println!("  total sum_z  = {total_z} (expected {})", THREADS as f32 * 3.0);
    // Each occupied slot must have exactly DUP observations.
    let per_slot_ok = (0..N_COORDS).all(|c| cs[slots[(c * DUP) as usize] as usize] == DUP);
    println!("  every occupied slot has exactly {DUP} observations = {per_slot_ok}");

    let t2 = total_count == exp_count
        && (total_w - exp_w).abs() < 1e-3
        && (total_x - THREADS as f32).abs() < 1e-2
        && (total_z - THREADS as f32 * 3.0).abs() < 1e-2
        && per_slot_ok;
    println!("  => Test 2 {}\n", if t2 { "PASS" } else { "FAIL" });
    pass &= t2;

    println!(
        "=== SPIKE RESULT: {} ===",
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        std::process::exit(1);
    }
}
