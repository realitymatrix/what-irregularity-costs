fn main() {
    // libcuda ships with the driver, not the toolkit, so it lives in the
    // driver stub dir when the toolkit is installed without a runtime driver.
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stubs");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rerun-if-changed=build.rs");
}
