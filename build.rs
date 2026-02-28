// Build script for janus-sim
// Adds library search path for libcufft_wrapper.so

fn main() {
    // Add target/release to library search path
    println!("cargo:rustc-link-search=native=target/release");

    // Tell cargo to rerun if the wrapper library changes
    println!("cargo:rerun-if-changed=cuda/cufft_wrapper.cu");
    println!("cargo:rerun-if-changed=target/release/libcufft_wrapper.so");
}
