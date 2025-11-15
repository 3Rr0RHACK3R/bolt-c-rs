use std::sync::Arc;

use bolt_core::{
    Tensor,
    device::DeviceKind,
    dispatcher::{Dispatcher, KernelLayoutReq},
    dtype::DType,
    error::{Error, ExpectedOutputs, Result},
    op::{OpAttrs, OpKey, OpKind},
    runtime::Runtime,
};
use bolt_cpu::CpuRuntimeBuilderExt;

#[test]
fn dispatch_multi_surfaces_all_outputs_and_single_errors() -> Result<()> {
    let runtime = runtime_with_test_kernels()?;
    let input = runtime.tensor_from_slice(&[2usize], &[1.0f32, 2.0])?;

    let outputs = runtime.dispatch_multi(OpKind::Fill, &[input.clone()], &OpAttrs::None)?;
    assert_eq!(outputs.len(), 2);

    let err = match runtime.dispatch_single(OpKind::Fill, &[input.clone()], OpAttrs::None) {
        Ok(_) => panic!("dispatch_single should error on >1 outputs"),
        Err(err) => err,
    };
    match err {
        Error::KernelOutputMismatch {
            op,
            expected,
            actual,
        } => {
            assert_eq!(op, OpKind::Fill);
            assert!(matches!(expected, ExpectedOutputs::Exactly(1)));
            assert_eq!(actual, 2);
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
    Ok(())
}

#[test]
fn dispatch_multi_allows_zero_outputs() -> Result<()> {
    let runtime = runtime_with_test_kernels()?;
    let input = runtime.tensor_from_slice(&[2usize], &[1i32, 2])?;

    let outputs = runtime.dispatch_multi(OpKind::Fill, &[input.clone()], &OpAttrs::None)?;
    assert!(outputs.is_empty());

    let err = match runtime.dispatch_single(OpKind::Fill, &[input.clone()], OpAttrs::None) {
        Ok(_) => panic!("dispatch_single should error on zero outputs"),
        Err(err) => err,
    };
    match err {
        Error::KernelOutputMismatch {
            op,
            expected,
            actual,
        } => {
            assert_eq!(op, OpKind::Fill);
            assert!(matches!(expected, ExpectedOutputs::Exactly(1)));
            assert_eq!(actual, 0);
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
    Ok(())
}

fn runtime_with_test_kernels() -> Result<Arc<Runtime>> {
    let mut builder = Runtime::builder().with_cpu()?;
    register_test_kernels(builder.dispatcher_mut())?;
    builder.build()
}

fn register_test_kernels(dispatcher: &mut Dispatcher) -> Result<()> {
    let multi_key = OpKey {
        op: OpKind::Fill,
        device: DeviceKind::Cpu,
        dtype: DType::F32,
    };
    dispatcher.register(
        multi_key,
        KernelLayoutReq::GeneralStrided,
        Arc::new(|inputs, _| multi_output_kernel(inputs)),
    )?;

    let zero_key = OpKey {
        op: OpKind::Fill,
        device: DeviceKind::Cpu,
        dtype: DType::I32,
    };
    dispatcher.register(
        zero_key,
        KernelLayoutReq::GeneralStrided,
        Arc::new(|inputs, _| zero_output_kernel(inputs)),
    )?;
    Ok(())
}

fn multi_output_kernel(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(Error::Device("test multi kernel expects 1 input".into()));
    }
    let input = inputs[0].clone();
    Ok(vec![input.clone(), input])
}

fn zero_output_kernel(inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(Error::Device("test zero kernel expects 1 input".into()));
    }
    Ok(Vec::new())
}
