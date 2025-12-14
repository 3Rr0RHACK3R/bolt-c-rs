use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::layers::HasParams;
use bolt_nn::layers::Linear;
use bolt_nn::layers::ModelExt;
use bolt_nn::layers::Seq;
use bolt_nn::layers::linear;
use bolt_nn::layers::relu;
use bolt_nn::Eval;

type B = CpuBackend;
type D = f32;

#[test]
fn linear_visits_weight_then_bias() {
    let backend = Arc::new(CpuBackend::new());
    let layer: Linear<B, D> = linear(2, 3).build(&backend).unwrap();

    let expected = vec![layer.weight.id(), layer.bias.as_ref().unwrap().id()];

    let mut actual = Vec::new();
    layer.visit_params(&mut |p| actual.push(p.id()));

    assert_eq!(layer.param_count(), expected.len());
    assert_eq!(actual, expected);
}

#[test]
fn then_visits_left_then_right() {
    let backend = Arc::new(CpuBackend::new());

    let left: Linear<B, D> = linear(2, 2).bias(false).build(&backend).unwrap();
    let left_weight = left.weight.id();

    let right: Linear<B, D> = linear(2, 2).bias(false).build(&backend).unwrap();
    let right_weight = right.weight.id();

    let model = left.then(right);

    let mut actual = Vec::new();
    model.visit_params(&mut |p| actual.push(p.id()));

    assert_eq!(model.param_count(), 2);
    assert_eq!(actual, vec![left_weight, right_weight]);
}

#[test]
fn seq_visits_layers_in_insertion_order() {
    let backend = Arc::new(CpuBackend::new());

    let first: Linear<B, D> = linear(2, 2).bias(false).build(&backend).unwrap();
    let first_weight = first.weight.id();

    let last: Linear<B, D> = linear(2, 2).bias(false).build(&backend).unwrap();
    let last_weight = last.weight.id();

    let model: Seq<B, D, Eval<B, D>> = Seq::new().push(first).push(relu()).push(last);

    let mut actual = Vec::new();
    model.visit_params(&mut |p| actual.push(p.id()));

    assert_eq!(model.param_count(), 2);
    assert_eq!(actual, vec![first_weight, last_weight]);
}
