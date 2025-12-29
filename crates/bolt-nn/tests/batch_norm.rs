use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::layers::{BatchNorm, Linear, Seq};
use bolt_nn::{ForwardCtx, Module, Store};
use bolt_tensor::Tensor;

type B = CpuBackend;
type D = f32;

fn backend() -> Arc<B> {
    Arc::new(CpuBackend::new())
}

fn store(backend: &Arc<B>) -> Store<B, D> {
    Store::new(backend.clone(), 42)
}

#[test]
fn batch_norm_train_forward_runs() {
    let backend = backend();
    let store = store(&backend);
    let bn = BatchNorm::<B, D>::init_default(&store.sub("bn"), 3).unwrap();

    let x = Tensor::<B, D>::from_slice(
        &backend,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[4, 3],
    )
    .unwrap();

    let mut ctx = ForwardCtx::train();
    let y = bn.forward(x.clone(), &mut ctx).unwrap();

    assert_eq!(y.shape(), x.shape());
}

#[test]
fn batch_norm_eval_forward_runs() {
    let backend = backend();
    let store = store(&backend);
    let bn = BatchNorm::<B, D>::init_default(&store.sub("bn"), 2).unwrap();

    let x_train = Tensor::<B, D>::from_slice(
        &backend,
        &[0.0, 10.0, 2.0, 12.0, 4.0, 14.0, 6.0, 16.0],
        &[4, 2],
    )
    .unwrap();

    let mut train_ctx = ForwardCtx::train();
    let _ = bn.forward(x_train, &mut train_ctx).unwrap();

    let x_eval = Tensor::<B, D>::from_slice(&backend, &[3.0, 13.0], &[1, 2]).unwrap();

    let mut eval_ctx = ForwardCtx::eval();
    let y_eval = bn.forward(x_eval, &mut eval_ctx).unwrap();

    let y_vals = y_eval.to_vec().unwrap();
    assert_eq!(y_vals.len(), 2);
}

#[test]
fn batch_norm_output_normalized() {
    let backend = backend();
    let store = store(&backend);
    let bn = BatchNorm::<B, D>::init(&store.sub("bn"), 2, false, 1e-5, 0.1).unwrap();

    let x = Tensor::<B, D>::from_slice(
        &backend,
        &[1.0, 5.0, 3.0, 7.0, 5.0, 9.0, 7.0, 11.0],
        &[4, 2],
    )
    .unwrap();

    let mut ctx = ForwardCtx::train();
    let y = bn.forward(x, &mut ctx).unwrap();

    let y_vals = y.to_vec().unwrap();

    let mean_ch0 = (y_vals[0] + y_vals[2] + y_vals[4] + y_vals[6]) / 4.0;
    let mean_ch1 = (y_vals[1] + y_vals[3] + y_vals[5] + y_vals[7]) / 4.0;

    assert!(
        mean_ch0.abs() < 1e-5,
        "channel 0 mean should be ~0, got {}",
        mean_ch0
    );
    assert!(
        mean_ch1.abs() < 1e-5,
        "channel 1 mean should be ~0, got {}",
        mean_ch1
    );

    let var_ch0 =
        (y_vals[0].powi(2) + y_vals[2].powi(2) + y_vals[4].powi(2) + y_vals[6].powi(2)) / 4.0;
    let var_ch1 =
        (y_vals[1].powi(2) + y_vals[3].powi(2) + y_vals[5].powi(2) + y_vals[7].powi(2)) / 4.0;

    assert!(
        (var_ch0 - 1.0).abs() < 0.1,
        "channel 0 variance should be ~1, got {}",
        var_ch0
    );
    assert!(
        (var_ch1 - 1.0).abs() < 0.1,
        "channel 1 variance should be ~1, got {}",
        var_ch1
    );
}

#[test]
fn batch_norm_shape_validation() {
    let backend = backend();
    let store = store(&backend);
    let bn = BatchNorm::<B, D>::init_default(&store.sub("bn"), 4).unwrap();

    let x_wrong = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0], &[1, 3]).unwrap();

    let mut ctx = ForwardCtx::train();
    let result = bn.forward(x_wrong, &mut ctx);

    assert!(result.is_err());
}

#[test]
fn batch_norm_4d_input() {
    let backend = backend();
    let store = store(&backend);
    let bn = BatchNorm::<B, D>::init_default(&store.sub("bn"), 2).unwrap();

    let x = Tensor::<B, D>::from_slice(
        &backend,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        &[2, 2, 2, 2],
    )
    .unwrap();

    let mut ctx = ForwardCtx::train();
    let y = bn.forward(x.clone(), &mut ctx).unwrap();

    assert_eq!(y.shape(), x.shape());
}

#[test]
fn batch_norm_gradients_flow() {
    let backend = backend();
    let store = store(&backend);
    let _bn = BatchNorm::<B, D>::init_default(&store.sub("bn"), 2).unwrap();

    let trainable = store.trainable();
    let trainable_keys: Vec<_> = trainable.iter().map(|p| p.key().to_string()).collect();

    assert!(
        trainable_keys.iter().any(|k| k.contains("gamma")),
        "gamma should be trainable"
    );
    assert!(
        trainable_keys.iter().any(|k| k.contains("beta")),
        "beta should be trainable"
    );
}

#[test]
fn batch_norm_running_stats_not_trainable() {
    let backend = backend();
    let store = store(&backend);
    let _bn = BatchNorm::<B, D>::init_default(&store.sub("bn"), 2).unwrap();

    let trainable = store.trainable();
    let trainable_keys: Vec<_> = trainable.iter().map(|p| p.key().to_string()).collect();

    assert!(
        !trainable_keys.iter().any(|k| k.contains("running")),
        "running stats should not be trainable"
    );
}

#[test]
fn batch_norm_multiple_train_passes() {
    let backend = backend();
    let store = store(&backend);
    let bn = BatchNorm::<B, D>::init_default(&store.sub("bn"), 2).unwrap();

    let x1 = Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    let x2 = Tensor::<B, D>::from_slice(&backend, &[5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

    let mut ctx1 = ForwardCtx::train();
    let y1 = bn.forward(x1, &mut ctx1).unwrap();

    let mut ctx2 = ForwardCtx::train();
    let y2 = bn.forward(x2, &mut ctx2).unwrap();

    assert_eq!(y1.shape().as_slice(), &[2, 2]);
    assert_eq!(y2.shape().as_slice(), &[2, 2]);
}

#[test]
fn batch_norm_train_then_eval_deterministic() {
    let backend = backend();
    let store = store(&backend);
    let bn = BatchNorm::<B, D>::init_default(&store.sub("bn"), 2).unwrap();

    let x_train =
        Tensor::<B, D>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();

    let mut train_ctx = ForwardCtx::train();
    let _ = bn.forward(x_train, &mut train_ctx).unwrap();

    let x_eval = Tensor::<B, D>::from_slice(&backend, &[2.5, 3.5], &[1, 2]).unwrap();

    let mut eval_ctx1 = ForwardCtx::eval();
    let y1 = bn.forward(x_eval.clone(), &mut eval_ctx1).unwrap();

    let mut eval_ctx2 = ForwardCtx::eval();
    let y2 = bn.forward(x_eval, &mut eval_ctx2).unwrap();

    assert_eq!(
        y1.to_vec().unwrap(),
        y2.to_vec().unwrap(),
        "eval should be deterministic"
    );
}

#[test]
fn batch_norm_normalized_before_linear() {
    let backend = backend();
    let store = store(&backend);

    let num_features = 3;
    let batch_size = 4;

    let bn = BatchNorm::<B, D>::init_default(&store.sub("bn"), num_features).unwrap();

    let x = Tensor::<B, D>::from_slice(
        &backend,
        &[
            10.0, 20.0, 30.0, 15.0, 25.0, 35.0, 20.0, 30.0, 40.0, 25.0, 35.0, 45.0,
        ],
        &[batch_size, num_features],
    )
    .unwrap();

    let mut ctx = ForwardCtx::train();
    let y_normalized = bn.forward(x.clone(), &mut ctx).unwrap();

    let y_vals = y_normalized.to_vec().unwrap();

    for ch in 0..num_features {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for b in 0..batch_size {
            let idx = b * num_features + ch;
            let val = y_vals[idx];
            sum += val;
            sum_sq += val * val;
        }

        let mean = sum / batch_size as f32;
        let variance = (sum_sq / batch_size as f32) - (mean * mean);

        assert!(
            mean.abs() < 1e-4,
            "channel {} mean should be ~0, got {}",
            ch,
            mean
        );
        assert!(
            (variance - 1.0).abs() < 0.15,
            "channel {} variance should be ~1, got {}",
            ch,
            variance
        );
    }

    let bn_for_seq = BatchNorm::<B, D>::init_default(&store.sub("bn_seq"), num_features).unwrap();
    let linear = Linear::init(&store.sub("linear"), num_features, 2, true).unwrap();

    let model: Seq<B, D> = Seq::new().push(bn_for_seq).push(linear);

    let mut ctx2 = ForwardCtx::train();
    let y_final = model.forward(x, &mut ctx2).unwrap();

    assert_eq!(y_final.shape().as_slice(), &[batch_size, 2]);
}
