use std::{marker::PhantomData, sync::Arc};

use crate::{
    allocator::StorageAllocator,
    backend::Backend,
    dtype::{DType, NativeType, ToF32},
    error::{Error, Result},
    layout::Layout,
    shape::ConcreteShape,
};
use bytemuck::cast;

#[derive(Clone)]
pub struct Tensor<B, D>
where
    B: Backend<D>,
    D: NativeType,
{
    backend: Arc<B>,
    storage: B::Storage,
    layout: Layout,
    _marker: PhantomData<D>,
}

impl<B, D> Tensor<B, D>
where
    B: Backend<D>,
    D: NativeType,
{
    pub fn from_slice(backend: &Arc<B>, data: &[D], shape: &[usize]) -> Result<Self> {
        let shape = ConcreteShape::from_slice(shape)?;
        if shape.num_elements() != data.len() {
            return Err(Error::SizeMismatch {
                expected: shape.num_elements(),
                actual: data.len(),
            });
        }
        let layout = Layout::contiguous(shape);
        let mut storage = backend.allocator().allocate(data.len())?;
        backend.write(&mut storage, &layout, data)?;
        let tensor = Self::from_parts(backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn zeros(backend: &Arc<B>, shape: &[usize]) -> Result<Self> {
        let shape = ConcreteShape::from_slice(shape)?;
        let numel = shape.num_elements();
        let layout = Layout::contiguous(shape);
        let storage = backend.allocator().allocate_zeroed(numel)?;
        let tensor = Self::from_parts(backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn ones(backend: &Arc<B>, shape: &[usize]) -> Result<Self> {
        Self::full(backend, shape, Self::one_value())
    }

    pub fn full(backend: &Arc<B>, shape: &[usize], value: D) -> Result<Self> {
        let shape = ConcreteShape::from_slice(shape)?;
        let layout = Layout::contiguous(shape);
        let storage = backend.fill(&layout, value)?;
        let tensor = Self::from_parts(backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn ones_like(other: &Tensor<B, D>) -> Result<Self> {
        let layout = Layout::with_strides(
            other.layout.concrete_shape().clone(),
            other.layout.strides(),
            0,
        )?;
        let storage = other.backend.fill(&layout, Self::one_value())?;
        let tensor = Self::from_parts(other.backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn full_like(other: &Tensor<B, D>, value: D) -> Result<Self> {
        let layout = Layout::with_strides(
            other.layout.concrete_shape().clone(),
            other.layout.strides(),
            0,
        )?;
        let storage = other.backend.fill(&layout, value)?;
        let tensor = Self::from_parts(other.backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn arange(backend: &Arc<B>, start: D, end: D, step: D) -> Result<Self> {
        let len = Self::compute_arange_len(start, end, step)?;
        let shape = ConcreteShape::from_slice(&[len])?;
        let layout = Layout::contiguous(shape);
        let mut storage = backend.allocator().allocate(len)?;
        let values = Self::build_arange_values(len, start, step)?;
        backend.write(&mut storage, &layout, &values)?;
        let tensor = Self::from_parts(backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn backend(&self) -> Arc<B> {
        self.backend.clone()
    }

    pub fn device(&self) -> &B::Device {
        self.backend.device()
    }

    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn numel(&self) -> usize {
        self.layout.num_elements()
    }

    pub fn to_vec(&self) -> Result<Vec<D>> {
        let tensor = if self.layout.is_contiguous() && self.layout.offset_bytes() == 0 {
            self.clone()
        } else {
            self.contiguous()?
        };
        let mut values = vec![D::default(); tensor.numel()];
        tensor
            .backend
            .read(&tensor.storage, &tensor.layout, &mut values)?;
        Ok(values)
    }

    pub fn item(&self) -> Result<D> {
        if self.numel() != 1 {
            return Err(Error::invalid_shape(
                "item requires a tensor with exactly one element",
            ));
        }
        let mut value = [D::default(); 1];
        self.backend.read(&self.storage, &self.layout, &mut value)?;
        Ok(value[0])
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        let shape = ConcreteShape::from_slice(shape)?;
        let layout = self.layout.reshape(shape)?;
        self.validate_layout_for_storage(&self.storage, &layout)?;
        Ok(self.with_layout(layout))
    }

    pub fn slice(&self, axis: usize, start: usize, end: usize, step: usize) -> Result<Self> {
        let layout = self.layout.slice(axis, start, end, step, D::DTYPE)?;
        self.validate_layout_for_storage(&self.storage, &layout)?;
        Ok(self.with_layout(layout))
    }

    pub fn permute(&self, axes: &[usize]) -> Result<Self> {
        let layout = self.layout.permute(axes)?;
        self.validate_layout_for_storage(&self.storage, &layout)?;
        Ok(self.with_layout(layout))
    }

    pub fn transpose(&self, axis_a: usize, axis_b: usize) -> Result<Self> {
        let layout = self.layout.transpose(axis_a, axis_b)?;
        self.validate_layout_for_storage(&self.storage, &layout)?;
        Ok(self.with_layout(layout))
    }

    pub fn contiguous(&self) -> Result<Self> {
        if self.layout.is_contiguous() && self.layout.offset_bytes() == 0 {
            return Ok(self.clone());
        }
        let parts = self.backend.copy(&self.storage, &self.layout)?;
        self.validate_layout_for_storage(&parts.storage, &parts.layout)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        self.ensure_same_backend(other)?;
        let parts = self
            .backend
            .add(&self.storage, &other.storage, &self.layout, &other.layout)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn sub(&self, other: &Self) -> Result<Self> {
        self.ensure_same_backend(other)?;
        let parts = self
            .backend
            .sub(&self.storage, &other.storage, &self.layout, &other.layout)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn matmul(&self, other: &Self) -> Result<Self> {
        self.ensure_same_backend(other)?;
        let parts =
            self.backend
                .matmul(&self.storage, &other.storage, &self.layout, &other.layout)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn mean_f32(&self) -> Result<Tensor<B, f32>>
    where
        B: Backend<f32>,
        D: ToF32,
    {
        let parts = Backend::<D>::mean_f32(self.backend.as_ref(), &self.storage, &self.layout)?;
        Ok(Tensor::<B, f32>::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    fn ensure_same_backend(&self, other: &Self) -> Result<()> {
        if self.backend.device_kind() != other.backend.device_kind() {
            return Err(Error::Device("tensors belong to different devices".into()));
        }
        if !Arc::ptr_eq(&self.backend, &other.backend) {
            return Err(Error::Device(
                "tensors belong to different backend instances".into(),
            ));
        }
        Ok(())
    }

    fn validate_layout_for_storage(&self, storage: &B::Storage, layout: &Layout) -> Result<()> {
        let len_bytes = self.backend.storage_len_bytes(storage);
        layout.validate_bounds(D::DTYPE, len_bytes)
    }

    fn with_layout(&self, layout: Layout) -> Self {
        Self {
            backend: self.backend.clone(),
            storage: self.storage.clone(),
            layout,
            _marker: PhantomData,
        }
    }

    fn from_parts(backend: Arc<B>, storage: B::Storage, layout: Layout) -> Self {
        Self {
            backend,
            storage,
            layout,
            _marker: PhantomData,
        }
    }

    pub fn storage(&self) -> &B::Storage {
        &self.storage
    }

    fn one_value() -> D {
        match D::DTYPE {
            DType::F32 => cast(1.0f32),
            DType::F64 => cast(1.0f64),
            DType::I32 => cast(1i32),
        }
    }

    fn compute_arange_len(start: D, end: D, step: D) -> Result<usize> {
        match D::DTYPE {
            DType::F32 => Self::float_arange_len(
                cast::<D, f32>(start) as f64,
                cast::<D, f32>(end) as f64,
                cast::<D, f32>(step) as f64,
            )
            .and_then(|len| usize::try_from(len).map_err(|_| Error::TensorTooLarge {
                limit: isize::MAX as usize,
                requested: usize::MAX,
            })),
            DType::F64 => Self::float_arange_len(
                cast::<D, f64>(start),
                cast::<D, f64>(end),
                cast::<D, f64>(step),
            )
            .and_then(|len| usize::try_from(len).map_err(|_| Error::TensorTooLarge {
                limit: isize::MAX as usize,
                requested: usize::MAX,
            })),
            DType::I32 => Self::int_arange_len(
                cast::<D, i32>(start),
                cast::<D, i32>(end),
                cast::<D, i32>(step),
            ),
        }
    }

    fn float_arange_len(start: f64, end: f64, step: f64) -> Result<u128> {
        if step == 0.0 || step.is_nan() {
            return Err(Error::invalid_shape(
                "arange step must be non-zero and not NaN",
            ));
        }
        if start.is_nan() || end.is_nan() {
            return Err(Error::invalid_shape("arange start/end must not be NaN"));
        }
        let span = end - start;
        let len = (span / step).ceil();
        if !len.is_finite() || len <= 0.0 {
            return Err(Error::invalid_shape(
                "arange would produce zero or infinite elements",
            ));
        }
        if len > isize::MAX as f64 {
            return Err(Error::TensorTooLarge {
                limit: isize::MAX as usize,
                requested: usize::MAX,
            });
        }
        let len_int = len as i128;
        if len_int <= 0 {
            return Err(Error::invalid_shape("arange would produce zero elements"));
        }
        u128::try_from(len_int).map_err(|_| Error::TensorTooLarge {
            limit: isize::MAX as usize,
            requested: usize::MAX,
        })
    }

    fn int_arange_len(start: i32, end: i32, step: i32) -> Result<usize> {
        if step == 0 {
            return Err(Error::invalid_shape("arange step must be non-zero"));
        }
        let delta = (end as i64) - (start as i64);
        let step_i64 = step as i64;
        if delta == 0 {
            return Err(Error::invalid_shape("arange would produce zero elements"));
        }
        if (delta > 0 && step_i64 <= 0) || (delta < 0 && step_i64 >= 0) {
            return Err(Error::invalid_shape(
                "arange step does not progress toward end",
            ));
        }
        let abs_delta = delta.abs() as i128;
        let abs_step = step_i64.unsigned_abs() as i128;
        let len = (abs_delta + (abs_step - 1)) / abs_step;
        if len == 0 {
            return Err(Error::invalid_shape("arange would produce zero elements"));
        }
        usize::try_from(len).map_err(|_| Error::TensorTooLarge {
            limit: isize::MAX as usize,
            requested: usize::MAX,
        })
    }

    fn build_arange_values(len: usize, start: D, step: D) -> Result<Vec<D>> {
        match D::DTYPE {
            DType::F32 => {
                let start = cast::<D, f32>(start);
                let step = cast::<D, f32>(step);
                let mut values = Vec::with_capacity(len);
                for idx in 0..len {
                    values.push(cast(start + step * idx as f32));
                }
                Ok(values)
            }
            DType::F64 => {
                let start = cast::<D, f64>(start);
                let step = cast::<D, f64>(step);
                let mut values = Vec::with_capacity(len);
                for idx in 0..len {
                    values.push(cast(start + step * idx as f64));
                }
                Ok(values)
            }
            DType::I32 => {
                let start_i64 = cast::<D, i32>(start) as i64;
                let step_i64 = cast::<D, i32>(step) as i64;
                let mut current = start_i64;
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    let value = i32::try_from(current)
                        .map_err(|_| Error::invalid_shape("arange value overflows i32 range"))?;
                    values.push(cast(value));
                    current = current.checked_add(step_i64).ok_or_else(|| {
                        Error::invalid_shape("arange value overflow during iteration")
                    })?;
                }
                Ok(values)
            }
        }
    }
}
