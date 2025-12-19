use std::sync::Arc;

use bolt_autodiff::Parameter;
use bolt_core::Tensor;
use bolt_cpu::CpuBackend;
use bolt_nn::layers::{HasParams, Seq, linear};
use bolt_nn::state_dict::{load_state_dict, state_dict};

type B = CpuBackend;
type D = f32;

fn set_param(param: &mut Parameter<B, D>, backend: &Arc<B>, values: &[f32], shape: &[usize]) {
    *param.tensor_mut() = Tensor::<B, D>::from_slice(backend, values, shape).unwrap();
}

#[test]
fn seq_state_dict_roundtrip_restores_weights() {
    let backend = Arc::new(CpuBackend::new());

    let mut l1 = linear(2, 2).bias(false).build(&backend).unwrap();
    set_param(&mut l1.weight, &backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let mut l2 = linear(2, 1).bias(false).build(&backend).unwrap();
    set_param(&mut l2.weight, &backend, &[5.0, 6.0], &[1, 2]);

    let mut model: Seq<B, D> = Seq::new().push(l1).push(l2);

    let sd = state_dict::<B, D, _>(&model).unwrap();

    assert!(sd.contains_key("layers.0.p0.weight"));
    assert!(sd.contains_key("layers.1.p0.weight"));

    // Mutate params.
    let mut params = model.params_mut();
    set_param(params[0], &backend, &[9.0, 9.0, 9.0, 9.0], &[2, 2]);
    set_param(params[1], &backend, &[8.0, 8.0], &[1, 2]);

    let report = load_state_dict::<B, D, _>(&mut model, &backend, &sd, true).unwrap();
    assert!(report.missing_keys.is_empty());
    assert!(report.unexpected_keys.is_empty());
    assert!(report.shape_mismatches.is_empty());
    assert!(report.dtype_mismatches.is_empty());

    let params = model.params();
    let w1 = params[0].tensor().to_vec().unwrap();
    let w2 = params[1].tensor().to_vec().unwrap();

    assert_eq!(w1, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(w2, vec![5.0, 6.0]);
}
