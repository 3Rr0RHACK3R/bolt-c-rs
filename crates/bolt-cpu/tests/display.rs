use std::sync::Arc;

use bolt_core::Tensor;
use bolt_cpu::CpuBackend;

#[test]
fn display_formats_1d_int_tensor() {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::from_slice(&backend, &[1i32, 2, 3], &[3]).unwrap();

    let rendered = format!("{tensor}");

    assert_eq!(
        rendered,
        "tensor([1, 2, 3], shape=[3], dtype=i32, device=cpu)"
    );
}

#[test]
fn display_formats_2d_float_tensor_with_newlines() {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::from_slice(&backend, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

    let rendered = format!("{tensor}");

    assert_eq!(
        rendered,
        "tensor([[1, 2, 3],\n  [4, 5, 6]\n], shape=[2, 3], dtype=f32, device=cpu)"
    );
}

#[test]
fn display_truncates_large_tensor() {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<i32> = (0..1200).collect();
    let tensor = Tensor::from_slice(&backend, &data, &[1200]).unwrap();

    let rendered = format!("{tensor}");

    assert_eq!(
        rendered,
        "tensor([0, 1, 2, ..., 1197, 1198, 1199], shape=[1200], dtype=i32, device=cpu)"
    );
}

#[test]
fn display_respects_non_contiguous_layouts() {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::arange(&backend, 0i32, 6, 1).unwrap();
    let reshaped = tensor.reshape(&[2, 3]).unwrap();
    let transposed = reshaped.transpose(0, 1).unwrap();

    let rendered = format!("{transposed}");

    assert_eq!(
        rendered,
        "tensor([[0, 3],\n  [1, 4],\n  [2, 5]\n], shape=[3, 2], dtype=i32, device=cpu)"
    );
}
