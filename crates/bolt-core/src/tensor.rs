use std::sync::Arc;

use bytemuck::cast_slice;

use crate::{
    buffer::{BufferId, BufferView},
    device::{Device, DeviceKind},
    dtype::{DType, NativeType},
    error::{Error, Result},
    layout::Layout,
    op::{OpAttrs, OpKind},
    runtime::Runtime,
    shape::{ConcreteShape, canonical_axes},
};

#[derive(Clone)]
pub struct Tensor {
    storage: Arc<TensorStorage>,
    view: BufferView,
}

pub struct TensorStorage {
    runtime: Arc<Runtime>,
    device_kind: DeviceKind,
    buffer_id: BufferId,
}

impl Tensor {
    pub(crate) fn zeros_in(
        runtime: &Arc<Runtime>,
        device_kind: DeviceKind,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Self> {
        let tensor = Self::allocate_uninit(runtime, device_kind, shape, dtype)?;
        let zeros = vec![0u8; tensor.view.num_bytes()];
        let device = runtime.device(device_kind)?;
        device.write(tensor.view.buffer_id, tensor.view.offset_bytes(), &zeros)?;
        Ok(tensor)
    }

    pub(crate) fn from_slice_in<T: NativeType>(
        runtime: &Arc<Runtime>,
        device_kind: DeviceKind,
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
        let tensor =
            Self::allocate_with_shape(runtime, device_kind, shape.clone(), T::DTYPE)?;
        let bytes = cast_slice(data);
        let device = runtime.device(device_kind)?;
        device.write(tensor.view.buffer_id, tensor.view.offset_bytes(), bytes)?;
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
        self.device()?
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

    pub fn runtime(&self) -> Arc<Runtime> {
        self.storage.runtime.clone()
    }

    pub(crate) fn runtime_ptr(&self) -> &Arc<Runtime> {
        &self.storage.runtime
    }

    pub fn dtype(&self) -> DType {
        self.view.dtype
    }

    pub fn device_kind(&self) -> DeviceKind {
        self.storage.device_kind
    }

    pub fn device(&self) -> Result<Arc<dyn Device>> {
        self.storage.device()
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

    pub(crate) fn allocate_uninit(
        runtime: &Arc<Runtime>,
        device_kind: DeviceKind,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Self> {
        let shape = ConcreteShape::from_slice(shape)?;
        Self::allocate_with_shape(runtime, device_kind, shape, dtype)
    }

    fn allocate_with_shape(
        runtime: &Arc<Runtime>,
        device_kind: DeviceKind,
        shape: ConcreteShape,
        dtype: DType,
    ) -> Result<Self> {
        let len_bytes = shape.num_elements() * dtype.size_in_bytes();
        let storage =
            Arc::new(TensorStorage::new(runtime.clone(), device_kind, len_bytes, dtype)?);
        let layout = Layout::contiguous(shape.clone());
        let view = BufferView {
            buffer_id: storage.buffer_id,
            dtype,
            layout,
        };
        Ok(Self { storage, view })
    }

    fn ensure_same_device_dtype(&self, other: &Tensor) -> Result<()> {
        self.ensure_same_runtime(other)?;
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

    fn ensure_same_runtime(&self, other: &Tensor) -> Result<()> {
        if !Arc::ptr_eq(self.runtime_ptr(), other.runtime_ptr()) {
            return Err(Error::Device(
                "tensors belong to different runtimes".into(),
            ));
        }
        Ok(())
    }

    fn dispatch_single(&self, op: OpKind, inputs: &[Tensor]) -> Result<Tensor> {
        self.dispatch_with_attrs(op, inputs, OpAttrs::None)
    }

    fn dispatch_with_attrs(&self, op: OpKind, inputs: &[Tensor], attrs: OpAttrs) -> Result<Tensor> {
        let runtime = self.runtime();
        runtime.dispatch_single(op, inputs, attrs)
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
    pub fn new(
        runtime: Arc<Runtime>,
        device_kind: DeviceKind,
        len_bytes: usize,
        dtype: DType,
    ) -> Result<Self> {
        if len_bytes == 0 {
            return Err(Error::invalid_shape(
                "tensor must have at least one element",
            ));
        }
        let device = runtime.device(device_kind)?;
        let buffer_id = device.alloc(len_bytes, dtype.alignment())?;
        Ok(Self {
            runtime,
            device_kind,
            buffer_id,
        })
    }

    pub fn device(&self) -> Result<Arc<dyn Device>> {
        self.runtime.device(self.device_kind)
    }
}

impl Drop for TensorStorage {
    fn drop(&mut self) {
        match self.runtime.device(self.device_kind) {
            Ok(device) => {
                if let Err(err) = device.free(self.buffer_id) {
                    eprintln!("failed to release buffer {:?}: {err}", self.buffer_id.raw());
                }
            }
            Err(err) => {
                eprintln!(
                    "failed to recover device {:?} for buffer {:?}: {err}",
                    self.device_kind, self.buffer_id.raw()
                );
            }
        }
    }
}
