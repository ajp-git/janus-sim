// Build script for janus-pm
// Compiles CUDA kernels and links to CUDA libraries

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // CUDA library paths (standard locations)
    let cuda_paths = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12/lib64",
        "/usr/local/cuda-12.3/lib64",
        "/usr/lib/x86_64-linux-gnu",
    ];

    for path in &cuda_paths {
        println!("cargo:rustc-link-search=native={}", path);
    }

    // Link to CUDA runtime and CuFFT
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cufft");

    // Always compile CUDA kernels
    compile_cuda_kernels();

    // Rerun if these change
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=LD_LIBRARY_PATH");
    println!("cargo:rerun-if-changed=src/kernels/pm_kernels.cu");
}

fn compile_cuda_kernels() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let cuda_src = manifest_dir.join("src/kernels/pm_kernels.cu");
    let cuda_lib = out_dir.join("libpm_kernels.so");

    // Check if source exists
    if !cuda_src.exists() {
        println!("cargo:warning=CUDA kernel source not found: {:?}", cuda_src);
        return;
    }

    println!("cargo:warning=Compiling CUDA kernels: {:?}", cuda_src);

    // Compile with nvcc
    let output = Command::new("nvcc")
        .args(&[
            "-O3",
            "-arch=sm_86",           // RTX 3060 = Ampere sm_86
            "-shared",
            "-Xcompiler", "-fPIC",
            "-o", cuda_lib.to_str().unwrap(),
            cuda_src.to_str().unwrap(),
            "-lcufft",
            "-lcudart",
        ])
        .output()
        .expect("Failed to run nvcc. Is CUDA installed?");

    if !output.status.success() {
        eprintln!("nvcc stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("nvcc stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("nvcc compilation failed");
    }

    println!("cargo:warning=CUDA kernels compiled successfully to {:?}", cuda_lib);

    // Tell cargo to link the library
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=pm_kernels");
}
