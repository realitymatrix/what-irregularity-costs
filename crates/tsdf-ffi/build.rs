use std::env;

fn main() {
    // Arm A0 is the correctness oracle, so it is compiled from the vendored
    // sources verbatim rather than reimplemented. See vendor/README.md.
    //
    // Native sm_120 by default. Deliberately no PTX fallback: an arm that JITs
    // from PTX while another runs native SASS would invalidate the comparison,
    // so a wrong arch must fail loudly at load rather than silently JIT.
    let arch = env::var("TSDF_CUDA_ARCH").unwrap_or_else(|_| "sm_120".to_string());

    let cuda_include = env::var("CUDA_INCLUDE_DIR")
        .unwrap_or_else(|_| "/usr/local/cuda/include".to_string());

    println!("cargo:rerun-if-changed=vendor/tsdf.cu");
    println!("cargo:rerun-if-changed=vendor/tsdf.h");
    println!("cargo:rerun-if-changed=vendor/tsdf_hash.cu");
    println!("cargo:rerun-if-changed=vendor/tsdf_hash.h");
    println!("cargo:rerun-if-env-changed=TSDF_CUDA_ARCH");

    cc::Build::new()
        .cuda(true)
        // CRITICAL: device debug info must stay OFF.
        //
        // The workspace release profile sets `debug = true` to keep host
        // symbols for profiling. The `cc` crate honours that by passing `-G` to
        // nvcc, which disables *device* optimisation entirely. Measured on the
        // marching-cubes kernel: 56 registers with -G off, 138 with it on. At
        // BD3 = 512 threads that is 70,656 registers per block against a 65,536
        // limit, so the kernel fails to launch with "too many resources
        // requested for launch".
        //
        // The launch failure is the lucky outcome. A kernel that merely got
        // slower would have silently poisoned every timing number in the
        // project, and -G costs far more than the register count suggests.
        .debug(false)
        .file("vendor/tsdf.cu")
        .file("vendor/tsdf_hash.cu")
        .include(&cuda_include)
        .include("vendor")
        .flag("-O3")
        // Line info for ncu/compute-sanitizer source attribution. Unlike -G,
        // this does not disable optimisation.
        .flag("-lineinfo")
        .flag(format!("-arch={arch}"))
        .compile("tsdf_reference");

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
}
