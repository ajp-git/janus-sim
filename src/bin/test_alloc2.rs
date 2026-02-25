/// Test GPU memory allocation including htod_sync_copy
/// cargo run --release --features cuda --bin test_alloc2 -- 20000000

#[cfg(feature = "cuda")]
fn main() {
    use cudarc::driver::CudaDevice;

    let n: usize = std::env::args().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000_000);

    println!("Testing with htod_sync_copy for N = {} ({:.1}M)", n, n as f64 / 1e6);

    let device = CudaDevice::new(0).expect("Failed to init CUDA");

    // Step 0: Compile PTX (this uses GPU memory!)
    println!("\n0. Compiling PTX kernels...");
    let ptx_src = include_str!("../../src/nbody_gpu.rs");
    // Actually we need to extract just the CUDA code, but let's simulate with the real compilation
    // Just check memory before/after
    println!("   Skipping PTX compilation in test (done in real init)");

    // Step 1: Generate data on CPU
    println!("\n1. Generating {} particles on CPU...", n);
    let pos_data: Vec<f64> = (0..n*3).map(|i| i as f64 * 0.001).collect();
    let vel_data: Vec<f64> = (0..n*3).map(|i| i as f64 * 0.0001).collect();
    let signs_data: Vec<i32> = (0..n).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
    println!("   CPU vectors created:");
    println!("   pos_data: {} bytes ({:.1} MB)", pos_data.len() * 8, pos_data.len() as f64 * 8.0 / 1e6);
    println!("   vel_data: {} bytes ({:.1} MB)", vel_data.len() * 8, vel_data.len() as f64 * 8.0 / 1e6);
    println!("   signs_data: {} bytes ({:.1} MB)", signs_data.len() * 4, signs_data.len() as f64 * 4.0 / 1e6);

    // Step 2: Upload with htod_sync_copy
    println!("\n2. Uploading to GPU with htod_sync_copy...");

    print!("   pos... ");
    let pos = match device.htod_sync_copy(&pos_data) {
        Ok(p) => { println!("OK"); p }
        Err(e) => { println!("FAILED: {}", e); return; }
    };

    print!("   vel... ");
    let vel = match device.htod_sync_copy(&vel_data) {
        Ok(v) => { println!("OK"); v }
        Err(e) => { println!("FAILED: {}", e); return; }
    };

    print!("   signs... ");
    let signs = match device.htod_sync_copy(&signs_data) {
        Ok(s) => { println!("OK"); s }
        Err(e) => { println!("FAILED: {}", e); return; }
    };

    // Step 3: Allocate remaining buffers
    println!("\n3. Allocating remaining buffers...");
    let n_bvh = 2 * n - 1;

    print!("   acc... ");
    let _acc = match device.alloc_zeros::<f64>(n * 3) {
        Ok(a) => { println!("OK"); a }
        Err(e) => { println!("FAILED: {}", e); return; }
    };

    print!("   morton... ");
    let _morton = match device.alloc_zeros::<u64>(n) {
        Ok(m) => { println!("OK"); m }
        Err(e) => { println!("FAILED: {}", e); return; }
    };

    print!("   sorted_idx... ");
    let _sorted = match device.alloc_zeros::<i32>(n) {
        Ok(s) => { println!("OK"); s }
        Err(e) => { println!("FAILED: {}", e); return; }
    };

    print!("   pos_tmp... ");
    let _pos_tmp = match device.alloc_zeros::<f64>(n * 3) {
        Ok(p) => { println!("OK"); p }
        Err(e) => { println!("FAILED: {}", e); return; }
    };

    print!("   vel_tmp... ");
    let _vel_tmp = match device.alloc_zeros::<f64>(n * 3) {
        Ok(v) => { println!("OK"); v }
        Err(e) => { println!("FAILED: {}", e); return; }
    };

    print!("   signs_tmp... ");
    let _signs_tmp = match device.alloc_zeros::<i32>(n) {
        Ok(s) => { println!("OK"); s }
        Err(e) => { println!("FAILED: {}", e); return; }
    };

    println!("\n4. Allocating BVH buffers ({:.1}M nodes)...", n_bvh as f64 / 1e6);

    print!("   bvh_left... ");
    let _left = match device.alloc_zeros::<i32>(n_bvh) {
        Ok(l) => { println!("OK"); l }
        Err(e) => { println!("FAILED: {}", e); return; }
    };

    print!("   bvh_right... ");
    let _right = match device.alloc_zeros::<i32>(n_bvh) {
        Ok(r) => { println!("OK"); r }
        Err(e) => { println!("FAILED: {}", e); return; }
    };

    print!("   bvh_parent... ");
    let _parent = match device.alloc_zeros::<i32>(n_bvh) {
        Ok(p) => { println!("OK"); p }
        Err(e) => { println!("FAILED: {}", e); return; }
    };

    print!("   bvh_range_left... ");
    let _rl = match device.alloc_zeros::<i32>(n_bvh) {
        Ok(r) => { println!("OK"); r }
        Err(e) => { println!("FAILED: {}", e); return; }
    };

    print!("   bvh_range_right... ");
    let _rr = match device.alloc_zeros::<i32>(n_bvh) {
        Ok(r) => { println!("OK"); r }
        Err(e) => { println!("FAILED: {}", e); return; }
    };

    print!("   bvh_node_data ({}×12 f64 = {:.1} GB)... ", n_bvh, n_bvh as f64 * 12.0 * 8.0 / 1e9);
    let _node_data = match device.alloc_zeros::<f64>(n_bvh * 12) {
        Ok(d) => { println!("OK"); d }
        Err(e) => { println!("FAILED: {}", e); return; }
    };

    print!("   bvh_node_types... ");
    let _types = match device.alloc_zeros::<i32>(n_bvh) {
        Ok(t) => { println!("OK"); t }
        Err(e) => { println!("FAILED: {}", e); return; }
    };

    print!("   bvh_atomic... ");
    let _atomic = match device.alloc_zeros::<i32>(n_bvh) {
        Ok(a) => { println!("OK"); a }
        Err(e) => { println!("FAILED: {}", e); return; }
    };

    println!("\n==========================================");
    println!("SUCCESS! All allocations completed.");
    println!("==========================================");

    // Keep references alive
    drop(pos);
    drop(vel);
    drop(signs);
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA not enabled");
}
