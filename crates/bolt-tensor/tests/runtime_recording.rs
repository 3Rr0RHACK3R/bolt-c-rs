use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_tensor::{Tensor, backward, no_grad};

#[test]
fn backward_computes_grads_for_add_sum() {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .unwrap()
        .requires_grad();
    let b = Tensor::<CpuBackend, f32>::from_slice(&backend, &[10.0, 20.0, 30.0, 40.0], &[2, 2])
        .unwrap()
        .requires_grad();

    let y = a.add(&b).unwrap();
    let loss = y.sum(None, false).unwrap();

    let grads = backward(&loss).unwrap();
    let ga = grads.wrt(&a).unwrap().to_vec().unwrap();
    let gb = grads.wrt(&b).unwrap().to_vec().unwrap();

    assert_eq!(ga, vec![1.0; 4]);
    assert_eq!(gb, vec![1.0; 4]);
}

#[test]
fn backward_computes_grads_for_matmul_sum() {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .unwrap()
        .requires_grad();
    let b = Tensor::<CpuBackend, f32>::from_slice(&backend, &[10.0, 20.0, 30.0, 40.0], &[2, 2])
        .unwrap()
        .requires_grad();

    let y = a.matmul(&b).unwrap();
    let loss = y.sum(None, false).unwrap();

    let grads = backward(&loss).unwrap();
    let ga = grads.wrt(&a).unwrap().to_vec().unwrap();
    let gb = grads.wrt(&b).unwrap().to_vec().unwrap();

    assert_eq!(ga, vec![30.0, 70.0, 30.0, 70.0]);
    assert_eq!(gb, vec![4.0, 4.0, 6.0, 6.0]);
}

#[test]
fn backward_computes_grads_through_view_ops() {
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(
        &backend,
        &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 1, 3],
    )
    .unwrap()
    .requires_grad();

    let y = x.squeeze().unwrap();
    let z = y.unsqueeze(1).unwrap();
    let w = z.transpose(0, 2).unwrap();
    let loss = w.sum(None, false).unwrap();

    let grads = loss.backward().unwrap();
    let gx = grads.wrt(&x).unwrap().to_vec().unwrap();
    assert_eq!(gx, vec![1.0; 6]);
}

#[test]
fn backward_computes_grads_for_broadcast_to_sum() {
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0_f32, 2.0, 3.0], &[3])
        .unwrap()
        .requires_grad();
    let y = x.broadcast_to(&[2, 3]).unwrap();
    let loss = y.sum(None, false).unwrap();

    let grads = backward(&loss).unwrap();
    let gx = grads.wrt(&x).unwrap().to_vec().unwrap();
    assert_eq!(gx, vec![2.0, 2.0, 2.0]);
}

#[test]
fn no_grad_disables_recording_and_backward_errors() {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0], &[2])
        .unwrap()
        .requires_grad();
    let b = Tensor::<CpuBackend, f32>::from_slice(&backend, &[3.0, 4.0], &[2])
        .unwrap()
        .requires_grad();

    let _ng = no_grad();
    let y = a.add(&b).unwrap();
    let loss = y.sum(None, false).unwrap();

    assert!(backward(&loss).is_err());
}
