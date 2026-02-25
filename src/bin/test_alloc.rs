/// Test GPU memory allocation step by step
/// cargo run --release --features cuda --bin test_alloc -- 20000000

#[cfg(feature = "cuda")]
fn main() {
    use cudarc::driver::CudaDevice;

    let n: usize = std::env::args().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000_000);

    let n_bvh = 2 * n - 1;

    println!("Testing GPU allocations for N = {} ({:.1}M)", n, n as f64 / 1e6);
    println!("BVH nodes: {} ({:.1}M)", n_bvh, n_bvh as f64 / 1e6);

    let device = CudaDevice::new(0).expect("Failed to init CUDA");

    let mut total_mb: f64 = 0.0;

    macro_rules! alloc_test {
        ($name:expr, $size:expr, $type:ty) => {{
            let size_bytes = $size * std::mem::size_of::<$type>();
            let size_mb = size_bytes as f64 / (1024.0 * 1024.0);
            print!("  {} ({:.1} MB)... ", $name, size_mb);
            match device.alloc_zeros::<$type>($size) {
                Ok(_buf) => {
                    total_mb += size_mb;
                    println!("OK (total: {:.1} MB)", total_mb);
                    true
                }
                Err(e) => {
                    println!("FAILED: {}", e);
                    false
                }
            }
        }};
    }

    println!("\n1. PARTICLE DATA:");
    if !alloc_test!("pos", n * 3, f64) { return; }
    if !alloc_test!("vel", n * 3, f64) { return; }
    if !alloc_test!("acc", n * 3, f64) { return; }
    if !alloc_test!("signs", n, i32) { return; }

    println!("\n2. SORTING BUFFERS:");
    if !alloc_test!("morton_codes", n, u64) { return; }
    if !alloc_test!("sorted_indices", n, i32) { return; }
    if !alloc_test!("pos_tmp", n * 3, f64) { return; }
    if !alloc_test!("vel_tmp", n * 3, f64) { return; }
    if !alloc_test!("signs_tmp", n, i32) { return; }

    println!("\n3. BVH TREE:");
    if !alloc_test!("bvh_left_child", n_bvh, i32) { return; }
    if !alloc_test!("bvh_right_child", n_bvh, i32) { return; }
    if !alloc_test!("bvh_parent", n_bvh, i32) { return; }
    if !alloc_test!("bvh_range_left", n_bvh, i32) { return; }
    if !alloc_test!("bvh_range_right", n_bvh, i32) { return; }
    if !alloc_test!("bvh_node_data", n_bvh * 12, f64) { return; }
    if !alloc_test!("bvh_node_types", n_bvh, i32) { return; }
    if !alloc_test!("bvh_atomic", n_bvh, i32) { return; }

    println!("\n==========================================");
    println!("SUCCESS! Total allocated: {:.1} MB ({:.2} GB)", total_mb, total_mb / 1024.0);
    println!("==========================================");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA not enabled");
}
