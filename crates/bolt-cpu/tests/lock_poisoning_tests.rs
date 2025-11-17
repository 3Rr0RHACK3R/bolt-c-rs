#![cfg(feature = "test-kernels")]

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
};

use bolt_core::{
    device::DeviceKind,
    error::Error,
    op::{OpAttrs, OpKind},
    runtime::Runtime,
    tensor::Tensor,
};
use bolt_cpu::{CpuRuntimeBuilderExt, register_test_poison_kernel};

fn runtime_with_poison_kernel() -> Arc<Runtime> {
    let mut builder = Runtime::builder();
    builder = builder.with_cpu().expect("cpu registration succeeds");
    register_test_poison_kernel(builder.dispatcher_mut())
        .expect("test poison kernel registration succeeds");
    builder.build().expect("runtime build succeeds")
}

fn dispatch_poison(runtime: &Arc<Runtime>, tensor: &Tensor) {
    let runtime = Arc::clone(runtime);
    let tensor = tensor.clone();
    let panic_result = catch_unwind(AssertUnwindSafe(|| {
        let inputs = vec![tensor];
        runtime
            .dispatch_single(OpKind::Fill, &inputs, OpAttrs::None)
            .expect("poison dispatch panics before returning");
    }));
    assert!(
        panic_result.is_err(),
        "poison kernel must panic to induce lock poisoning"
    );
}

fn assert_lock_poisoned(err: Error, expected_lock: &str) {
    match err {
        Error::DeviceLockPoisoned { device, lock } => {
            assert_eq!(device, DeviceKind::Cpu);
            assert_eq!(lock, expected_lock);
        }
        other => panic!("expected DeviceLockPoisoned, got {other:?}"),
    }
}

#[test]
fn cpu_device_lock_poisoning_returns_error_instead_of_panic() {
    let runtime = runtime_with_poison_kernel();
    let tensor = runtime
        .tensor_from_slice(&[2], &[1.0f32, 2.0f32])
        .expect("tensor allocation succeeds");

    dispatch_poison(&runtime, &tensor);

    let err = tensor
        .to_vec::<f32>()
        .expect_err("poisoned buffer read should fail");
    assert_lock_poisoned(err, "buffer_cell.read");
}

#[test]
fn cpu_device_lock_poisoning_is_scoped_to_buffer() {
    let runtime = runtime_with_poison_kernel();
    let tensor_a = runtime
        .tensor_from_slice(&[2], &[3.0f32, 4.0f32])
        .expect("tensor allocation succeeds");
    let tensor_b = runtime
        .tensor_from_slice(&[2], &[5.0f32, 6.0f32])
        .expect("tensor allocation succeeds");

    dispatch_poison(&runtime, &tensor_a);

    let err = tensor_a
        .to_vec::<f32>()
        .expect_err("poisoned buffer read should fail");
    assert_lock_poisoned(err, "buffer_cell.read");

    let values = tensor_b
        .to_vec::<f32>()
        .expect("independent buffer should remain accessible");
    assert_eq!(values, vec![5.0f32, 6.0f32]);
}
