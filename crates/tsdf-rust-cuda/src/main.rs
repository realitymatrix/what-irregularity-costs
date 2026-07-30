//! Smoke entry point for arm A4. The arm itself is driven through the C ABI in
//! `lib.rs`; this only confirms the device module loads on the target GPU.

fn main() {
    println!("=== A4 (Rust CUDA via cuda-oxide) ===");
    if tsdf_rust_cuda::a4_init() == 0 {
        println!("  device module loaded; kernels: alloc_kernel, update_kernel");
    } else {
        eprintln!("  failed to load device module");
        std::process::exit(1);
    }
}
