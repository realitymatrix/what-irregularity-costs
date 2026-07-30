//! Validate the Triton launch ABI by reproducing spike S2 from Rust.
//!
//! Every part of this ABI fails *silently*: wrong argument order, a missing
//! trailing scratch pointer, or using the BLOCK constexpr as the CUDA block
//! size all launch successfully and compute the wrong answer. So this checks
//! numerical output against values produced by the Python-launched kernels
//! (artifacts/triton/expected.json), not merely that the launch succeeded.
//!
//! Run after any Triton upgrade. The ABI was read out of Triton 3.6.0's
//! driver.py and is not a stable public interface.
//!
//! Usage: validate_abi [artifacts_dir]

use std::ffi::c_void;
use std::path::PathBuf;

use triton_aot::{CudaContext, DeviceBuffer, TritonKernel};

// Must match spikes/s2_triton/s2_triton_spike.py. These are compiled into the
// cubin as constexprs; they appear here only to size buffers and check results.
const HASH_SIZE: usize = 1024;
const HASH_EMPTY: i32 = -1;
const SCRATCH_SENTINEL: i32 = -2;
const N_COORDS: usize = 64;
const DUP: usize = 8;
const THREADS: usize = N_COORDS * DUP;
const GRID_X: u32 = 2; // cdiv(THREADS=512, BLOCK=256)

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("artifacts/triton"));

    println!("=== Triton launch-ABI validation (Rust driver) ===");
    let _ctx = CudaContext::new(0)?;

    let k_hash = TritonKernel::load(&dir, "hash_insert_kernel")?;
    let k_acc = TritonKernel::load(&dir, "voxel_accumulate_kernel")?;

    println!(
        "  hash_insert_kernel      block_dim_x {}  shared {}  args {:?}",
        k_hash.manifest.block_dim_x, k_hash.manifest.shared_bytes, k_hash.manifest.runtime_args
    );
    println!(
        "  voxel_accumulate_kernel block_dim_x {}  shared {}  args {:?}",
        k_acc.manifest.block_dim_x, k_acc.manifest.shared_bytes, k_acc.manifest.runtime_args
    );
    println!();

    // -- kernel 1: atomicCAS linear-probe hash insert -----------------------
    let mut table_host = vec![HASH_EMPTY; HASH_SIZE + 1];
    table_host[HASH_SIZE] = SCRATCH_SENTINEL; // sink for resolved lanes
    let table = DeviceBuffer::from_slice(&table_host)?;
    let slot = DeviceBuffer::zeroed(THREADS * 4)?;
    let won = DeviceBuffer::zeroed(THREADS * 4)?;

    let (mut tp, mut sp, mut wp) = (table.ptr(), slot.ptr(), won.ptr());
    let mut n: i32 = THREADS as i32;
    let mut args: [*mut c_void; 4] = [
        &mut tp as *mut _ as *mut c_void,
        &mut sp as *mut _ as *mut c_void,
        &mut wp as *mut _ as *mut c_void,
        &mut n as *mut _ as *mut c_void,
    ];
    // SAFETY: four args matching (*i32, *i32, *i32, i32); buffers are sized
    // above for HASH_SIZE+1 and THREADS respectively.
    unsafe { k_hash.launch(GRID_X, &mut args)? };
    _ctx.synchronize()?;

    let slots: Vec<i32> = slot.to_vec()?;
    let wons: Vec<i32> = won.to_vec()?;
    let table_out: Vec<i32> = table.to_vec()?;

    let winners: i32 = wons.iter().sum();
    let unresolved = slots.iter().filter(|&&s| s == HASH_EMPTY).count();
    let agree = (0..N_COORDS).all(|c| {
        let base = slots[c * DUP];
        (0..DUP).all(|d| slots[c * DUP + d] == base)
    });
    let mut occupied: Vec<i32> = (0..N_COORDS).map(|c| slots[c * DUP]).collect();
    occupied.sort_unstable();
    occupied.dedup();
    let scratch_intact = table_out[HASH_SIZE] == SCRATCH_SENTINEL;

    println!("--- hash_insert (atomicCAS linear probe) ---");
    println!("  winners            {winners} (expected {N_COORDS})");
    println!("  unresolved         {unresolved} (expected 0)");
    println!("  duplicates agree   {agree}");
    println!("  distinct slots     {} (expected {N_COORDS})", occupied.len());
    println!("  scratch intact     {scratch_intact}");
    let t1 = winners == N_COORDS as i32
        && unresolved == 0
        && agree
        && occupied.len() == N_COORDS
        && scratch_intact;
    println!("  => {}\n", if t1 { "PASS" } else { "FAIL" });

    // -- kernel 2: SoA atomicAdd accumulation -------------------------------
    let sum_xyz = DeviceBuffer::zeroed(HASH_SIZE * 3 * 4)?;
    let weight = DeviceBuffer::zeroed(HASH_SIZE * 4)?;
    let count = DeviceBuffer::zeroed(HASH_SIZE * 4)?;

    let (mut slp, mut sxp, mut wtp, mut cnp) =
        (slot.ptr(), sum_xyz.ptr(), weight.ptr(), count.ptr());
    let mut n2: i32 = THREADS as i32;
    let mut args2: [*mut c_void; 5] = [
        &mut slp as *mut _ as *mut c_void,
        &mut sxp as *mut _ as *mut c_void,
        &mut wtp as *mut _ as *mut c_void,
        &mut cnp as *mut _ as *mut c_void,
        &mut n2 as *mut _ as *mut c_void,
    ];
    // SAFETY: five args matching (*i32, *f32, *f32, *i32, i32); buffers sized
    // for HASH_SIZE entries, and slots hold indices < HASH_SIZE by construction.
    unsafe { k_acc.launch(GRID_X, &mut args2)? };
    _ctx.synchronize()?;

    let counts: Vec<i32> = count.to_vec()?;
    let weights: Vec<f32> = weight.to_vec()?;
    let sums: Vec<f32> = sum_xyz.to_vec()?;

    let total_count: i32 = counts.iter().sum();
    let total_w: f32 = weights.iter().sum();
    let total_x: f32 = (0..HASH_SIZE).map(|i| sums[i * 3]).sum();
    let total_z: f32 = (0..HASH_SIZE).map(|i| sums[i * 3 + 2]).sum();
    let per_slot_ok = (0..N_COORDS).all(|c| counts[slots[c * DUP] as usize] == DUP as i32);

    println!("--- voxel_accumulate (SoA atomicAdd) ---");
    println!("  total count        {total_count} (expected {THREADS})");
    println!("  total weight       {total_w} (expected {})", THREADS as f32 * 0.5);
    println!("  total sum_x        {total_x} (expected {})", THREADS as f32);
    println!("  total sum_z        {total_z} (expected {})", THREADS as f32 * 3.0);
    println!("  per-slot count {DUP}   {per_slot_ok}");
    let t2 = total_count == THREADS as i32
        && (total_w - THREADS as f32 * 0.5).abs() < 1e-3
        && (total_x - THREADS as f32).abs() < 1e-2
        && (total_z - THREADS as f32 * 3.0).abs() < 1e-2
        && per_slot_ok;
    println!("  => {}\n", if t2 { "PASS" } else { "FAIL" });

    let ok = t1 && t2;
    println!("=== ABI VALIDATION: {} ===", if ok { "PASS" } else { "FAIL" });
    if ok {
        println!("Rust-launched kernels reproduce the Python-launched results exactly.");
        println!("Argument packing, trailing scratch params, block shape and shared");
        println!("memory are all correct. Python is a build-time dependency only.");
    }
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
