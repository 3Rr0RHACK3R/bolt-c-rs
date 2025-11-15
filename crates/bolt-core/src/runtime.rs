use std::{collections::HashMap, sync::Arc};

use crate::{
    device::{Device, DeviceKind},
    dispatcher::Dispatcher,
    dtype::{DType, NativeType},
    error::{Error, ExpectedOutputs, Result},
    op::{OpAttrs, OpKey, OpKind, Operation},
    tensor::Tensor,
};

pub struct Runtime {
    dispatcher: Dispatcher,
    devices: HashMap<DeviceKind, Arc<dyn Device>>,
    default_device: DeviceKind,
}

impl Runtime {
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::new()
    }

    pub fn dispatcher(&self) -> &Dispatcher {
        &self.dispatcher
    }

    pub fn device(&self, kind: DeviceKind) -> Result<Arc<dyn Device>> {
        self.devices
            .get(&kind)
            .cloned()
            .ok_or_else(|| Error::Device(format!("device {kind:?} not registered")))
    }

    pub fn default_device(&self) -> DeviceKind {
        self.default_device
    }

    pub fn tensor_from_slice<T: NativeType>(
        self: &Arc<Self>,
        shape: &[usize],
        data: &[T],
    ) -> Result<Tensor> {
        self.tensor_from_slice_on(self.default_device, shape, data)
    }

    pub fn tensor_from_slice_on<T: NativeType>(
        self: &Arc<Self>,
        device_kind: DeviceKind,
        shape: &[usize],
        data: &[T],
    ) -> Result<Tensor> {
        Tensor::from_slice_in(self, device_kind, shape, data)
    }

    pub fn tensor_zeros(self: &Arc<Self>, shape: &[usize], dtype: DType) -> Result<Tensor> {
        self.tensor_zeros_on(self.default_device, shape, dtype)
    }

    pub fn tensor_zeros_on(
        self: &Arc<Self>,
        device_kind: DeviceKind,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor> {
        Tensor::zeros_in(self, device_kind, shape, dtype)
    }

    pub fn allocate_uninit(
        self: &Arc<Self>,
        device_kind: DeviceKind,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor> {
        Tensor::allocate_uninit(self, device_kind, shape, dtype)
    }

    pub fn dispatch_single(
        self: &Arc<Self>,
        op: OpKind,
        inputs: &[Tensor],
        attrs: OpAttrs,
    ) -> Result<Tensor> {
        let outputs = self.dispatch_multi(op, inputs, &attrs)?;
        if outputs.len() != 1 {
            return Err(Error::KernelOutputMismatch {
                op,
                expected: ExpectedOutputs::Exactly(1),
                actual: outputs.len(),
            });
        }
        Ok(outputs.into_iter().next().expect("len validated"))
    }

    pub fn dispatch_op<O>(self: &Arc<Self>, op: &O, inputs: &[Tensor]) -> Result<Vec<Tensor>>
    where
        O: Operation,
    {
        let attrs = op.to_opattrs();
        self.dispatch_multi(O::KIND, inputs, &attrs)
    }

    pub fn dispatch_op_single<O>(self: &Arc<Self>, op: &O, inputs: &[Tensor]) -> Result<Tensor>
    where
        O: Operation,
    {
        let outputs = self.dispatch_op(op, inputs)?;
        if outputs.len() != 1 {
            return Err(Error::KernelOutputMismatch {
                op: O::KIND,
                expected: ExpectedOutputs::Exactly(1),
                actual: outputs.len(),
            });
        }
        Ok(outputs.into_iter().next().expect("len validated"))
    }

    pub fn dispatch_multi(
        self: &Arc<Self>,
        op: OpKind,
        inputs: &[Tensor],
        attrs: &OpAttrs,
    ) -> Result<Vec<Tensor>> {
        self.ensure_inputs_belong(inputs)?;
        let first = inputs
            .first()
            .ok_or_else(|| Error::Device("dispatch requires at least one input".into()))?;
        let key = OpKey {
            op,
            device: first.device_kind(),
            dtype: first.dtype(),
        };
        self.dispatcher.dispatch(key, inputs, attrs)
    }

    fn ensure_inputs_belong(self: &Arc<Self>, inputs: &[Tensor]) -> Result<()> {
        for tensor in inputs {
            if !Arc::ptr_eq(tensor.runtime_ptr(), self) {
                return Err(Error::Device("inputs belong to different runtimes".into()));
            }
        }
        Ok(())
    }
}

pub struct RuntimeBuilder {
    dispatcher: Dispatcher,
    devices: HashMap<DeviceKind, Arc<dyn Device>>,
    default_device: Option<DeviceKind>,
}

impl RuntimeBuilder {
    pub fn new() -> Self {
        Self {
            dispatcher: Dispatcher::new(),
            devices: HashMap::new(),
            default_device: None,
        }
    }

    pub fn with_device(mut self, kind: DeviceKind, device: Arc<dyn Device>) -> Self {
        self.devices.insert(kind, device);
        self
    }

    pub fn with_default_device(mut self, kind: DeviceKind) -> Self {
        self.default_device = Some(kind);
        self
    }

    pub fn dispatcher_mut(&mut self) -> &mut Dispatcher {
        &mut self.dispatcher
    }

    pub fn build(self) -> Result<Arc<Runtime>> {
        let default_device = self
            .default_device
            .ok_or_else(|| Error::Device("default device not set".into()))?;

        if !self.devices.contains_key(&default_device) {
            return Err(Error::Device(
                "default device not present in device map".into(),
            ));
        }

        Ok(Arc::new(Runtime {
            dispatcher: self.dispatcher,
            devices: self.devices,
            default_device,
        }))
    }
}
