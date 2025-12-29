use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::layers::GlobalAvgPool;
use bolt_nn::{ForwardCtx, Module};
use bolt_tensor::Tensor;

type B = CpuBackend;
type D = f32;

fn assert_close(actual: f32, expected: f32, tol: f32, msg: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff < tol,
        "{}: expected {}, got {}, diff {}",
        msg,
        expected,
        actual,
        diff
    );
}

#[test]
fn global_avg_pool_4d_to_4d_shape() {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let input = Tensor::<B, D>::from_slice(&backend, &data, &[2, 3, 2, 2]).unwrap();

    let layer = GlobalAvgPool::new(true);
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();
    assert_eq!(output.shape().as_slice(), &[2, 3, 1, 1]);
}

#[test]
fn global_avg_pool_computes_correct_mean() {
    let backend = Arc::new(CpuBackend::new());
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input = Tensor::<B, D>::from_slice(&backend, &data, &[1, 2, 2, 2]).unwrap();

    let layer = GlobalAvgPool::new(true);
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();
    let result = output.to_vec().unwrap();

    assert_eq!(result.len(), 2);
    assert_close(result[0], 2.5, 1e-5, "channel 0 mean");
    assert_close(result[1], 6.5, 1e-5, "channel 1 mean");
}

#[test]
fn global_avg_pool_batch_independence() {
    let backend = Arc::new(CpuBackend::new());
    let data = vec![1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0];
    let input = Tensor::<B, D>::from_slice(&backend, &data, &[2, 1, 2, 2]).unwrap();

    let layer = GlobalAvgPool::new(true);
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();
    let result = output.to_vec().unwrap();

    assert_eq!(result.len(), 2);
    assert_close(result[0], 2.5, 1e-5, "batch 0 mean");
    assert_close(result[1], 10.5, 1e-5, "batch 1 mean");
}

#[test]
fn global_avg_pool_multiple_channels() {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
    let input = Tensor::<B, D>::from_slice(&backend, &data, &[1, 3, 4, 4]).unwrap();

    let layer = GlobalAvgPool::new(true);
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();
    let result = output.to_vec().unwrap();

    assert_eq!(result.len(), 3);
    assert_eq!(output.shape().as_slice(), &[1, 3, 1, 1]);

    let channel_0_mean = (0..16).sum::<usize>() as f32 / 16.0;
    let channel_1_mean = (16..32).sum::<usize>() as f32 / 16.0;
    let channel_2_mean = (32..48).sum::<usize>() as f32 / 16.0;

    assert_close(result[0], channel_0_mean, 1e-5, "channel 0");
    assert_close(result[1], channel_1_mean, 1e-5, "channel 1");
    assert_close(result[2], channel_2_mean, 1e-5, "channel 2");
}

#[test]
fn global_avg_pool_large_spatial_dims() {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (0..200).map(|i| i as f32).collect();
    let input = Tensor::<B, D>::from_slice(&backend, &data, &[1, 2, 10, 10]).unwrap();

    let layer = GlobalAvgPool::new(true);
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();

    assert_eq!(output.shape().as_slice(), &[1, 2, 1, 1]);
    let result = output.to_vec().unwrap();
    assert_eq!(result.len(), 2);

    let channel_0_mean = (0..100).sum::<usize>() as f32 / 100.0;
    let channel_1_mean = (100..200).sum::<usize>() as f32 / 100.0;

    assert_close(result[0], channel_0_mean, 1e-5, "channel 0");
    assert_close(result[1], channel_1_mean, 1e-5, "channel 1");
}

#[test]
#[should_panic(expected = "GlobalAvgPool expects at least 3D input")]
fn global_avg_pool_rejects_2d_input() {
    let backend = Arc::new(CpuBackend::new());
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let input = Tensor::<B, D>::from_slice(&backend, &data, &[2, 2]).unwrap();

    let layer = GlobalAvgPool::new(true);
    let mut ctx = ForwardCtx::eval();
    let _ = layer.forward(input, &mut ctx).unwrap();
}

#[test]
fn global_avg_pool_gradient_flow() {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let input = Tensor::<B, D>::from_slice(&backend, &data, &[1, 2, 2, 2])
        .unwrap()
        .requires_grad();

    let layer = GlobalAvgPool::new(true);
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input.clone(), &mut ctx).unwrap();

    let loss = output.sum(None, false).unwrap();
    let grads = loss.backward().unwrap();

    let grad = grads.wrt(&input).unwrap();
    assert_eq!(grad.shape().as_slice(), &[1, 2, 2, 2]);
}

#[test]
fn global_avg_pool_3d_input_keepdim() {
    let backend = Arc::new(CpuBackend::new());
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Tensor::<B, D>::from_slice(&backend, &data, &[1, 2, 3]).unwrap();

    let layer = GlobalAvgPool::new(true);
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();

    assert_eq!(output.shape().as_slice(), &[1, 2, 1]);
    let result = output.to_vec().unwrap();
    assert_close(result[0], 2.0, 1e-5, "channel 0 mean (1+2+3)/3");
    assert_close(result[1], 5.0, 1e-5, "channel 1 mean (4+5+6)/3");
}

#[test]
fn global_avg_pool_3d_input_no_keepdim() {
    let backend = Arc::new(CpuBackend::new());
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Tensor::<B, D>::from_slice(&backend, &data, &[1, 2, 3]).unwrap();

    let layer = GlobalAvgPool::new(false);
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();

    assert_eq!(output.shape().as_slice(), &[1, 2]);
    let result = output.to_vec().unwrap();
    assert_close(result[0], 2.0, 1e-5, "channel 0 mean");
    assert_close(result[1], 5.0, 1e-5, "channel 1 mean");
}

#[test]
fn global_avg_pool_4d_input_no_keepdim() {
    let backend = Arc::new(CpuBackend::new());
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input = Tensor::<B, D>::from_slice(&backend, &data, &[1, 2, 2, 2]).unwrap();

    let layer = GlobalAvgPool::new(false);
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();

    assert_eq!(output.shape().as_slice(), &[1, 2]);
    let result = output.to_vec().unwrap();
    assert_close(result[0], 2.5, 1e-5, "channel 0 mean");
    assert_close(result[1], 6.5, 1e-5, "channel 1 mean");
}

#[test]
fn global_avg_pool_5d_input_keepdim() {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let input = Tensor::<B, D>::from_slice(&backend, &data, &[1, 2, 2, 2, 3]).unwrap();

    let layer = GlobalAvgPool::new(true);
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();

    assert_eq!(output.shape().as_slice(), &[1, 2, 1, 1, 1]);
}

#[test]
fn global_avg_pool_5d_input_no_keepdim() {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let input = Tensor::<B, D>::from_slice(&backend, &data, &[1, 2, 2, 2, 3]).unwrap();

    let layer = GlobalAvgPool::new(false);
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input, &mut ctx).unwrap();

    assert_eq!(output.shape().as_slice(), &[1, 2]);
}

#[test]
fn global_avg_pool_6d_input() {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
    let input = Tensor::<B, D>::from_slice(&backend, &data, &[1, 2, 2, 2, 2, 3]).unwrap();

    let layer = GlobalAvgPool::new(true);
    let mut ctx = ForwardCtx::eval();
    let output = layer.forward(input.clone(), &mut ctx).unwrap();

    assert_eq!(output.shape().as_slice(), &[1, 2, 1, 1, 1, 1]);

    let layer_no_keepdim = GlobalAvgPool::new(false);
    let output_no_keepdim = layer_no_keepdim.forward(input, &mut ctx).unwrap();
    assert_eq!(output_no_keepdim.shape().as_slice(), &[1, 2]);
}
