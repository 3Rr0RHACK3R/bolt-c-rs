use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_tensor::{Tensor, backward};

#[test]
fn concat_along_axis_0() {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::from_slice(&backend, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::from_slice(&backend, &[5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

    let c = Tensor::concat(&[a, b], 0).unwrap();
    assert_eq!(c.shape().as_slice(), &[4, 2]);
    let data = c.to_vec().unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn concat_along_axis_1() {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::from_slice(&backend, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::from_slice(&backend, &[5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

    let c = Tensor::concat(&[a, b], 1).unwrap();
    assert_eq!(c.shape().as_slice(), &[2, 4]);
    let data = c.to_vec().unwrap();
    assert_eq!(data, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
}

#[test]
fn concat_single_tensor() {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::from_slice(&backend, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let c = Tensor::concat(&[a], 0).unwrap();
    assert_eq!(c.shape().as_slice(), &[2, 2]);
    let data = c.to_vec().unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn concat_three_tensors() {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::from_slice(&backend, &[1.0f32, 2.0], &[1, 2]).unwrap();
    let b = Tensor::from_slice(&backend, &[3.0f32, 4.0], &[1, 2]).unwrap();
    let c = Tensor::from_slice(&backend, &[5.0f32, 6.0], &[1, 2]).unwrap();

    let d = Tensor::concat(&[a, b, c], 0).unwrap();
    assert_eq!(d.shape().as_slice(), &[3, 2]);
    let data = d.to_vec().unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn concat_empty_list() {
    let _backend = Arc::new(CpuBackend::new());

    let result = Tensor::<CpuBackend, f32>::concat(&[], 0);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("empty"));
}

#[test]
fn concat_shape_mismatch() {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::from_slice(&backend, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::from_slice(&backend, &[5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0], &[2, 3]).unwrap();

    let result = Tensor::concat(&[a, b], 0);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("dimension mismatch"));
}

#[test]
fn concat_rank_mismatch() {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::from_slice(&backend, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::from_slice(&backend, &[5.0f32, 6.0, 7.0, 8.0], &[4]).unwrap();

    let result = Tensor::concat(&[a, b], 0);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("rank"));
}

#[test]
fn concat_axis_out_of_bounds() {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::from_slice(&backend, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::from_slice(&backend, &[5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

    let result = Tensor::concat(&[a, b], 2);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("out of bounds"));
}

#[test]
fn concat_1d_tensors() {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::from_slice(&backend, &[1.0f32, 2.0], &[2]).unwrap();
    let b = Tensor::from_slice(&backend, &[3.0f32, 4.0], &[2]).unwrap();

    let c = Tensor::concat(&[a, b], 0).unwrap();
    assert_eq!(c.shape().as_slice(), &[4]);
    let data = c.to_vec().unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn concat_3d_tensors() {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::from_slice(&backend, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).unwrap();
    let b = Tensor::from_slice(&backend, &[9.0f32, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], &[2, 2, 2]).unwrap();

    let c = Tensor::concat(&[a, b], 0).unwrap();
    assert_eq!(c.shape().as_slice(), &[4, 2, 2]);
}

#[test]
fn concat_non_contiguous() {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::from_slice(&backend, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::from_slice(&backend, &[5.0f32, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

    // Create a view (non-contiguous)
    let a_view = a.transpose(0, 1).unwrap();
    let b_view = b.transpose(0, 1).unwrap();

    let c = Tensor::concat(&[a_view, b_view], 0).unwrap();
    assert_eq!(c.shape().as_slice(), &[4, 2]);
}

#[test]
fn concat_gradient_flow() {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::from_slice(&backend, &[1.0f32, 2.0], &[1, 2])
        .unwrap()
        .requires_grad();
    let b = Tensor::from_slice(&backend, &[3.0f32, 4.0], &[1, 2])
        .unwrap()
        .requires_grad();

    let c = Tensor::concat(&[a.clone(), b.clone()], 0).unwrap();
    let loss = c.sum(None, false).unwrap();

    let grads = backward(&loss).unwrap();
    let ga = grads.wrt(&a).unwrap().to_vec().unwrap();
    let gb = grads.wrt(&b).unwrap().to_vec().unwrap();

    // Gradients should be all ones (sum of all ones)
    assert_eq!(ga, vec![1.0, 1.0]);
    assert_eq!(gb, vec![1.0, 1.0]);
}

#[test]
fn concat_gradient_flow_axis_1() {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::from_slice(&backend, &[1.0f32, 2.0], &[2, 1])
        .unwrap()
        .requires_grad();
    let b = Tensor::from_slice(&backend, &[3.0f32, 4.0], &[2, 1])
        .unwrap()
        .requires_grad();

    let c = Tensor::concat(&[a.clone(), b.clone()], 1).unwrap();
    let loss = c.sum(None, false).unwrap();

    let grads = backward(&loss).unwrap();
    let ga = grads.wrt(&a).unwrap().to_vec().unwrap();
    let gb = grads.wrt(&b).unwrap().to_vec().unwrap();

    assert_eq!(ga, vec![1.0, 1.0]);
    assert_eq!(gb, vec![1.0, 1.0]);
}

#[test]
fn concat_with_no_grad() {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::from_slice(&backend, &[1.0f32, 2.0], &[1, 2]).unwrap();
    let b = Tensor::from_slice(&backend, &[3.0f32, 4.0], &[1, 2]).unwrap();

    let c = Tensor::concat(&[a, b], 0).unwrap();
    // Concat with non-requires_grad tensors should work fine
    assert_eq!(c.shape().as_slice(), &[2, 2]);
}

#[test]
fn concat_integration_with_add() {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::from_slice(&backend, &[1.0f32, 2.0], &[1, 2]).unwrap();
    let b = Tensor::from_slice(&backend, &[3.0f32, 4.0], &[1, 2]).unwrap();
    let c = Tensor::from_slice(&backend, &[5.0f32, 6.0], &[1, 2]).unwrap();

    let concat_result = Tensor::concat(&[a, b], 0).unwrap();
    let add_result = concat_result.add(&c).unwrap();

    assert_eq!(add_result.shape().as_slice(), &[2, 2]);
}
