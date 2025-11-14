use std::sync::Arc;

use bytemuck::cast_slice;

use crate::{
    buffer::{BufferId, BufferView},
    device::{Device, DeviceKind},
    dispatcher::global_dispatcher,
    dtype::{DType, NativeType},
    error::{Error, Result},
    layout::Layout,
    op::{OpAttrs, OpKey, OpKind},
    shape::{ConcreteShape, canonical_axes},
};

#[derive(Clone)]
pub struct Tensor {
    storage: Arc<TensorStorage>,
    view: BufferView,
}

pub struct TensorStorage {
    device: Arc<dyn Device>,
    buffer_id: BufferId,
}

impl Tensor {
    pub fn zeros(device: Arc<dyn Device>, shape: &[usize], dtype: DType) -> Result<Self> {
        let tensor = Self::allocate_uninit(device, shape, dtype)?;
        let zeros = vec![0u8; tensor.view.num_bytes()];
        tensor
            .storage
            .device
            .write(tensor.view.buffer_id, tensor.view.offset_bytes(), &zeros)?;
        Ok(tensor)
    }

    pub fn from_slice<T: NativeType>(
        device: Arc<dyn Device>,
        shape: &[usize],
        data: &[T],
    ) -> Result<Self> {
        let shape = ConcreteShape::from_slice(shape)?;
        if shape.num_elements() != data.len() {
            return Err(Error::SizeMismatch {
                expected: shape.num_elements(),
                actual: data.len(),
            });
        }
        let tensor = Self::allocate_with_shape(device, shape.clone(), T::DTYPE)?;
        let bytes = cast_slice(data);
        tensor
            .storage
            .device
            .write(tensor.view.buffer_id, tensor.view.offset_bytes(), bytes)?;
        Ok(tensor)
    }

    pub fn to_vec<T: NativeType>(&self) -> Result<Vec<T>> {
        if self.view.dtype != T::DTYPE {
            return Err(Error::DTypeMismatch {
                lhs: self.view.dtype,
                rhs: T::DTYPE,
            });
        }
        if !self.is_contiguous() {
            return self.contiguous()?.to_vec();
        }
        let mut bytes = vec![0u8; self.view.num_bytes()];
        self.storage
            .device
            .read(self.view.buffer_id, self.view.offset_bytes(), &mut bytes)?;
        let values: Vec<T> = cast_slice(&bytes).to_vec();
        Ok(values)
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        let shape = ConcreteShape::from_slice(new_shape)?;
        let layout = self.view.layout.reshape(shape)?;
        Ok(self.with_layout(layout))
    }

    pub fn slice(&self, axis: usize, start: usize, end: usize, step: usize) -> Result<Self> {
        let layout = self
            .view
            .layout
            .slice(axis, start, end, step, self.view.dtype)?;
        Ok(self.with_layout(layout))
    }

    pub fn permute(&self, axes: &[usize]) -> Result<Self> {
        let layout = self.view.layout.permute(axes)?;
        Ok(self.with_layout(layout))
    }

    pub fn transpose(&self, axis_a: usize, axis_b: usize) -> Result<Self> {
        let layout = self.view.layout.transpose(axis_a, axis_b)?;
        Ok(self.with_layout(layout))
    }

    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        self.dispatch_single(OpKind::Copy, &[self.clone()])
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        self.ensure_same_device_dtype(other)?;
        self.dispatch_single(OpKind::Add, &[self.clone(), other.clone()])
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        self.ensure_same_device_dtype(other)?;
        self.dispatch_single(OpKind::Sub, &[self.clone(), other.clone()])
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        self.ensure_same_device_dtype(other)?;
        self.dispatch_single(OpKind::Mul, &[self.clone(), other.clone()])
    }

    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        self.ensure_same_device_dtype(other)?;
        self.dispatch_single(OpKind::Div, &[self.clone(), other.clone()])
    }

    pub fn neg(&self) -> Result<Tensor> {
        self.dispatch_single(OpKind::Neg, &[self.clone()])
    }

    pub fn exp(&self) -> Result<Tensor> {
        self.require_float("exp")?;
        self.dispatch_single(OpKind::Exp, &[self.clone()])
    }

    pub fn relu(&self) -> Result<Tensor> {
        self.require_float("relu")?;
        self.dispatch_single(OpKind::Relu, &[self.clone()])
    }

    pub fn sum(&self) -> Result<Tensor> {
        self.sum_axes(&[])
    }

    pub fn sum_axes(&self, axes: &[usize]) -> Result<Tensor> {
        let axes = canonical_axes(axes, self.shape().len())?;
        let attrs = OpAttrs::reduce(axes.clone());
        self.dispatch_with_attrs(OpKind::Sum, &[self.clone()], attrs)
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        self.ensure_same_device_dtype(other)?;
        if self.shape().len() != 2 {
            return Err(Error::invalid_shape(format!(
                "matmul lhs must be 2D, got {:?}",
                self.shape()
            )));
        }
        if other.shape().len() != 2 {
            return Err(Error::invalid_shape(format!(
                "matmul rhs must be 2D, got {:?}",
                other.shape()
            )));
        }
        let k_lhs = self.shape()[1];
        let k_rhs = other.shape()[0];
        if k_lhs != k_rhs {
            return Err(Error::ShapeMismatch {
                lhs: self.shape().to_vec(),
                rhs: other.shape().to_vec(),
            });
        }
        self.dispatch_single(OpKind::MatMul, &[self.clone(), other.clone()])
    }

    pub fn dtype(&self) -> DType {
        self.view.dtype
    }

    pub fn device_kind(&self) -> DeviceKind {
        self.storage.device.kind()
    }

    pub fn device(&self) -> Arc<dyn Device> {
        self.storage.device.clone()
    }

    pub fn shape(&self) -> &[usize] {
        self.view.layout.shape()
    }

    pub fn strides(&self) -> &[isize] {
        self.view.layout.strides()
    }

    pub fn layout(&self) -> &Layout {
        &self.view.layout
    }

    pub fn view(&self) -> &BufferView {
        &self.view
    }

    pub fn numel(&self) -> usize {
        self.view.layout.num_elements()
    }

    pub fn buffer_id(&self) -> BufferId {
        self.view.buffer_id
    }

    pub fn offset_bytes(&self) -> usize {
        self.view.layout.offset_bytes()
    }

    pub fn is_contiguous(&self) -> bool {
        self.view.layout.is_contiguous()
    }

    pub fn allocate_uninit(device: Arc<dyn Device>, shape: &[usize], dtype: DType) -> Result<Self> {
        let shape = ConcreteShape::from_slice(shape)?;
        Self::allocate_with_shape(device, shape, dtype)
    }

    fn allocate_with_shape(
        device: Arc<dyn Device>,
        shape: ConcreteShape,
        dtype: DType,
    ) -> Result<Self> {
        let len_bytes = shape.num_elements() * dtype.size_in_bytes();
        let storage = Arc::new(TensorStorage::new(device, len_bytes, dtype)?);
        let layout = Layout::contiguous(shape.clone());
        let view = BufferView {
            buffer_id: storage.buffer_id,
            dtype,
            layout,
        };
        Ok(Self { storage, view })
    }

    fn ensure_same_device_dtype(&self, other: &Tensor) -> Result<()> {
        if self.dtype() != other.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: other.dtype(),
            });
        }
        if self.device_kind() != other.device_kind() {
            return Err(Error::DeviceMismatch {
                lhs: self.device_kind(),
                rhs: other.device_kind(),
            });
        }
        Ok(())
    }

    fn dispatch_single(&self, op: OpKind, inputs: &[Tensor]) -> Result<Tensor> {
        self.dispatch_with_attrs(op, inputs, OpAttrs::None)
    }

    fn dispatch_with_attrs(&self, op: OpKind, inputs: &[Tensor], attrs: OpAttrs) -> Result<Tensor> {
        let dispatcher = global_dispatcher()?;
        let key = OpKey {
            op,
            device: self.device_kind(),
            dtype: self.dtype(),
        };
        let mut outputs = dispatcher.dispatch(key, inputs, &attrs)?;
        outputs
            .pop()
            .ok_or_else(|| Error::Device("kernel returned no outputs".into()))
    }

    fn require_float(&self, op: &'static str) -> Result<()> {
        if !self.dtype().is_float() {
            return Err(Error::RequiresFloat {
                op,
                dtype: self.dtype(),
            });
        }
        Ok(())
    }

    fn with_layout(&self, layout: Layout) -> Self {
        Self {
            storage: self.storage.clone(),
            view: BufferView {
                buffer_id: self.view.buffer_id,
                dtype: self.view.dtype,
                layout,
            },
        }
    }
}

impl TensorStorage {
    pub fn new(device: Arc<dyn Device>, len_bytes: usize, dtype: DType) -> Result<Self> {
        if len_bytes == 0 {
            return Err(Error::invalid_shape(
                "tensor must have at least one element",
            ));
        }
        let buffer_id = device.alloc(len_bytes, dtype.alignment())?;
        Ok(Self { device, buffer_id })
    }

    pub fn device(&self) -> &Arc<dyn Device> {
        &self.device
    }
}

impl Drop for TensorStorage {
    fn drop(&mut self) {
        if let Err(err) = self.device.free(self.buffer_id) {
            eprintln!("failed to release buffer {:?}: {err}", self.buffer_id.raw());
        }
    }
}
