use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::layers::BatchNorm;
use bolt_nn::{ForwardCtx, Module, Store};
use bolt_tensor::Tensor;

type B = CpuBackend;
type D = f32;

fn backend() -> Arc<B> {
    Arc::new(CpuBackend::new())
}

fn store(backend: &Arc<B>) -> Store<B, D> {
    Store::new(backend.clone(), 1337)
}

// This test specifies a distinct gamma per channel and checks that, in eval mode
// with running_mean=0 and running_var=1 (eps=0), the affine is applied along the
// channel axis (dim=1) for 4D inputs [N, C, H, W]. A buggy axis expansion that
// aligns to the last axis instead will not satisfy the per-channel ratio.
#[test]
fn batch_norm_affine_broadcasts_along_channel_axis_4d() {
    let backend = backend();
    let store = store(&backend);
    let bn = BatchNorm::<B, D>::init(&store.sub("bn"), /*C=*/3, /*affine=*/true, /*eps=*/0.0, 0.1).unwrap();

    // Set gamma = [1.0, 2.0, 3.0]
    let gamma_vals = [1.0_f32, 2.0, 3.0];
    for p in store.trainable() {
        if p.key().ends_with(".gamma") {
            let t = Tensor::<B, D>::from_slice(&backend, &gamma_vals, &[3]).unwrap();
            p.set_tensor(t).unwrap();
        }
    }

    // 4D input [N=2, C=3, H=2, W=2] with values 1..=24
    let x_vals: Vec<f32> = (1..=24).map(|v| v as f32).collect();
    let x = Tensor::<B, D>::from_vec(&backend, x_vals.clone(), &[2, 3, 2, 2]).unwrap();

    let mut ctx = ForwardCtx::eval();
    let y = bn.forward(x.clone(), &mut ctx).unwrap();
    let y_vals = y.to_vec().unwrap();

    // Verify per-channel ratio y/x equals gamma[c]
    let mut idx = 0;
    for _n in 0..2 {
        for c in 0..3 {
            for _h in 0..2 {
                for _w in 0..2 {
                    let xv = x_vals[idx];
                    let yv = y_vals[idx];
                    let ratio = yv / xv;
                    let expected = gamma_vals[c];
                    assert!(
                        (ratio - expected).abs() < 1e-5,
                        "expected channel {} ratio {}, got {} at flat idx {}",
                        c,
                        expected,
                        ratio,
                        idx
                    );
                    idx += 1;
                }
            }
        }
    }
}

// Same assertion for a 3D input [N, C, L] to catch axis alignment bugs beyond 4D case.
#[test]
fn batch_norm_affine_broadcasts_along_channel_axis_3d() {
    let backend = backend();
    let store = store(&backend);
    let bn = BatchNorm::<B, D>::init(&store.sub("bn3"), /*C=*/2, /*affine=*/true, /*eps=*/0.0, 0.1).unwrap();

    // Set gamma = [1.5, 0.5]
    let gamma_vals = [1.5_f32, 0.5];
    for p in store.trainable() {
        if p.key().ends_with(".gamma") && p.key().starts_with("bn3") {
            let t = Tensor::<B, D>::from_slice(&backend, &gamma_vals, &[2]).unwrap();
            p.set_tensor(t).unwrap();
        }
    }

    // 3D input [N=2, C=2, L=3] with values 1..=12
    let x_vals: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let x = Tensor::<B, D>::from_vec(&backend, x_vals.clone(), &[2, 2, 3]).unwrap();

    let mut ctx = ForwardCtx::eval();
    let y = bn.forward(x.clone(), &mut ctx).unwrap();
    let y_vals = y.to_vec().unwrap();

    // Verify per-channel ratio y/x equals gamma[c]
    let mut idx = 0;
    for _n in 0..2 {
        for c in 0..2 {
            for _l in 0..3 {
                let xv = x_vals[idx];
                let yv = y_vals[idx];
                let ratio = yv / xv;
                let expected = gamma_vals[c];
                assert!(
                    (ratio - expected).abs() < 1e-5,
                    "expected channel {} ratio {}, got {} at flat idx {}",
                    c,
                    expected,
                    ratio,
                    idx
                );
                idx += 1;
            }
        }
    }
}
