use bolt_core::{
    allocator::StorageAllocator,
    backend::Backend,
    dtype::DType,
};
use bolt_cpu::backend::CpuBackend;

#[test]
fn test_caching_allocator_reuse() {
    let backend = CpuBackend::with_pooling();
    // Specify we want the f32 allocator
    let allocator = <CpuBackend as Backend<f32>>::allocator(&backend);

    // 1. Allocate 1024 floats (4KB)
    let len_bytes = 1024 * 4;
    let t1 = allocator.allocate_bytes(len_bytes, DType::F32).unwrap();
    let ptr1 = unsafe { t1.as_slice().as_ptr() };

    // 2. Drop t1. It should go to pool.
    drop(t1);

    // 3. Allocate t2 with same size.
    let t2 = allocator.allocate_bytes(len_bytes, DType::F32).unwrap();
    let ptr2 = unsafe { t2.as_slice().as_ptr() };

    // 4. Assert pointers are same (reuse)
    assert_eq!(ptr1, ptr2, "Memory should be reused from the pool");
}

#[test]
fn test_caching_allocator_alignment() {
    let backend = CpuBackend::with_pooling();
    let allocator = <CpuBackend as Backend<f32>>::allocator(&backend);
    
    // Allocate small size, but check alignment
    let t1 = allocator.allocate(1).unwrap(); // 1 float
    let ptr = unsafe { t1.as_slice().as_ptr() } as usize;
    
    // We enforced 64-byte alignment in StorageBlock
    assert_eq!(ptr % 64, 0, "Pointer should be 64-byte aligned for SIMD");
}

#[test]
fn test_caching_allocator_multiple_sizes() {
    let backend = CpuBackend::with_pooling();
    let allocator = <CpuBackend as Backend<f32>>::allocator(&backend);

    let t1 = allocator.allocate(100).unwrap();
    let p1 = unsafe { t1.as_slice().as_ptr() };
    
    let t2 = allocator.allocate(200).unwrap();
    let p2 = unsafe { t2.as_slice().as_ptr() };

    drop(t1);
    
    // Request 100 again -> should get p1
    let t3 = allocator.allocate(100).unwrap();
    let p3 = unsafe { t3.as_slice().as_ptr() };
    assert_eq!(p1, p3);

    // Request 200 again -> should NOT get p1 (size mismatch)
    // But wait, t2 is still alive.
    drop(t2);
    let t4 = allocator.allocate(200).unwrap();
    let p4 = unsafe { t4.as_slice().as_ptr() };
    assert_eq!(p2, p4);
}