use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::layers::{Linear, Seq};
use bolt_nn::{LoadOptions, Store};
use bolt_tensor::Tensor;

type B = CpuBackend;
type D = f32;

fn set_param(param: &bolt_nn::Param<B, D>, backend: &Arc<B>, values: &[f32], shape: &[usize]) {
    let t = Tensor::<B, D>::from_slice(backend, values, shape).unwrap();
    param.set_tensor(t).unwrap();
}

#[test]
fn seq_state_dict_roundtrip_restores_weights() {
    let backend = Arc::new(CpuBackend::new());
    let store = Store::<B, D>::new(backend.clone(), 1337);

    let l1 = Linear::init(&store.sub("layers").sub_idx(0).sub("p0"), 2, 2, false).unwrap();
    set_param(&l1.weight, &backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let l2 = Linear::init(&store.sub("layers").sub_idx(1).sub("p0"), 2, 1, false).unwrap();
    set_param(&l2.weight, &backend, &[5.0, 6.0], &[1, 2]);

    let _model: Seq<B, D> = Seq::new().push(l1).push(l2);

    let sd = store.state_dict().unwrap();
    assert!(sd.tensors.contains_key("layers.0.p0.weight"));
    assert!(sd.tensors.contains_key("layers.1.p0.weight"));

    // TODO: maybe a helper on store for "updating/mutating" a param?
    set_param(
        &store
            .named_trainable()
            .into_iter()
            .find(|(k, _)| k == "layers.0.p0.weight")
            .unwrap()
            .1,
        &backend,
        &[9.0, 9.0, 9.0, 9.0],
        &[2, 2],
    );
    set_param(
        &store
            .named_trainable()
            .into_iter()
            .find(|(k, _)| k == "layers.1.p0.weight")
            .unwrap()
            .1,
        &backend,
        &[8.0, 8.0],
        &[1, 2],
    );

    let report = store
        .load_state_dict(
            &sd,
            LoadOptions {
                strict: true,
                rename: None,
            },
        )
        .unwrap();
    assert!(report.missing.is_empty());
    assert!(report.unexpected.is_empty());
    assert!(report.mismatched.is_empty());

    let w1 = store
        .named_trainable()
        .into_iter()
        .find(|(k, _)| k == "layers.0.p0.weight")
        .unwrap()
        .1
        .tensor()
        .to_vec()
        .unwrap();
    let w2 = store
        .named_trainable()
        .into_iter()
        .find(|(k, _)| k == "layers.1.p0.weight")
        .unwrap()
        .1
        .tensor()
        .to_vec()
        .unwrap();

    assert_eq!(w1, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(w2, vec![5.0, 6.0]);
}
