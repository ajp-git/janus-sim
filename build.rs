// Build script for janus-sim
// Adds library search paths for native libraries

fn main() {
    // Add target/release to library search path
    println!("cargo:rustc-link-search=native=target/release");

    // Tell cargo to rerun if the wrapper library changes
    println!("cargo:rerun-if-changed=cuda/cufft_wrapper.cu");
    println!("cargo:rerun-if-changed=target/release/libcufft_wrapper.so");

    // Grackle cooling library
    #[cfg(feature = "grackle")]
    {
        // Add Grackle library search paths
        println!("cargo:rustc-link-search=native=/usr/local/lib");
        println!("cargo:rustc-link-search=native=external/grackle-dist");
        println!("cargo:rustc-link-search=native=/app/external/grackle-dist");

        // Link Grackle libraries
        println!("cargo:rustc-link-lib=dylib=grackle_bridge");
        println!("cargo:rustc-link-lib=dylib=grackle");

        // Rerun if Grackle sources change
        println!("cargo:rerun-if-changed=external/grackle_bridge.c");
        println!("cargo:rerun-if-changed=external/grackle_bridge.h");
    }
}
