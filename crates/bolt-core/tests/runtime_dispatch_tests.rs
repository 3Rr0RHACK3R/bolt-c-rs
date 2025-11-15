use std::sync::Arc;

use bolt_core::{
    Tensor,
    device::{Device, DeviceKind},
    dispatcher::{Dispatcher, KernelLayoutReq},
    dtype::DType,
    error::{Error, ExpectedOutputs, Result},
    op::{OpAttrs, OpKey, OpKind, Operation},
    runtime::Runtime,
};
use bolt_cpu::{CpuDevice, CpuRuntimeBuilderExt};

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

#[test]
fn register_operation_surfaces_attr_mismatches() -> Result<()> {
    let runtime = runtime_with_typed_fill_kernel()?;
    let input = runtime.tensor_from_slice(&[2usize], &[1.0f32, 2.0])?;

    let mismatch_attrs = OpAttrs::Sum { axes: vec![0] };
    let err = match runtime.dispatch_multi(OpKind::Fill, &[input.clone()], &mismatch_attrs) {
        Ok(_) => panic!("mismatched attrs should error"),
        Err(err) => err,
    };
    match err {
        Error::OpAttrMismatch {
            op,
            expected,
            actual,
        } => {
            assert_eq!(op, OpKind::Fill);
            assert_eq!(expected, "None");
            assert_eq!(actual, "Sum");
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

#[derive(Clone, Debug)]
struct TestFillOp;

impl Operation for TestFillOp {
    const KIND: OpKind = OpKind::Fill;

    fn to_opattrs(&self) -> OpAttrs {
        OpAttrs::None
    }

    fn from_opattrs(attrs: &OpAttrs) -> Result<Self> {
        match attrs {
            OpAttrs::None => Ok(TestFillOp),
            other => Err(Error::OpAttrMismatch {
                op: OpKind::Fill,
                expected: "None",
                actual: other.discriminant_name(),
            }),
        }
    }
}

fn runtime_with_typed_fill_kernel() -> Result<Arc<Runtime>> {
    let mut builder = Runtime::builder();
    {
        let dispatcher = builder.dispatcher_mut();
        dispatcher.register_operation::<TestFillOp, _>(
            DeviceKind::Cpu,
            DType::F32,
            KernelLayoutReq::GeneralStrided,
            |inputs, _op| {
                if inputs.len() != 1 {
                    return Err(Error::Device(
                        "typed fill test kernel expects 1 input".into(),
                    ));
                }
                Ok(vec![inputs[0].clone()])
            },
        )?;
    }
    let device = Arc::new(CpuDevice::new()) as Arc<dyn Device>;
    builder = builder.with_device(DeviceKind::Cpu, device);
    builder = builder.with_default_device(DeviceKind::Cpu);
    builder.build()
}
