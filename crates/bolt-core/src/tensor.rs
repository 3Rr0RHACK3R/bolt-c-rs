use std::{marker::PhantomData, sync::Arc};

use crate::{
    allocator::StorageAllocator,
    backend::{
        AbsOp, AddOp, Backend, CopyOp, CosOp, DivOp, ExpOp, FillOp, LogOp, MatmulOp, MeanOp, MulOp,
        NegOp, PowOp, ReluOp, SinOp, SqrtOp, SubOp, TanhOp,
    },
    dtype::{FloatType, NativeType, OneValue, ToF32},
    error::{Error, Result},
    index::TensorIndex,
    layout::Layout,
    shape::ConcreteShape,
    utils::tensor_creation,
};

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

    pub fn from_vec(backend: &Arc<B>, data: Vec<D>, shape: &[usize]) -> Result<Self> {
        let shape = ConcreteShape::from_slice(shape)?;
        if shape.num_elements() != data.len() {
            return Err(Error::SizeMismatch {
                expected: shape.num_elements(),
                actual: data.len(),
            });
        }
        let layout = Layout::contiguous(shape);
        let mut storage = backend.allocator().allocate(data.len())?;
        backend.write(&mut storage, &layout, &data)?;
        let tensor = Self::from_parts(backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn zeros(backend: &Arc<B>, shape: &[usize]) -> Result<Self>
    where
        B: FillOp<D>,
    {
        let shape = ConcreteShape::from_slice(shape)?;
        let numel = shape.num_elements();
        let layout = Layout::contiguous(shape);
        let storage = backend.allocator().allocate_zeroed(numel)?;
        let tensor = Self::from_parts(backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn ones(backend: &Arc<B>, shape: &[usize]) -> Result<Self>
    where
        D: OneValue,
        B: FillOp<D>,
    {
        Self::full(backend, shape, tensor_creation::one_value::<D>())
    }

    pub fn full(backend: &Arc<B>, shape: &[usize], value: D) -> Result<Self>
    where
        B: FillOp<D>,
    {
        let shape = ConcreteShape::from_slice(shape)?;
        let layout = Layout::contiguous(shape);
        let storage = backend.fill(&layout, value)?;
        let tensor = Self::from_parts(backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn ones_like(other: &Tensor<B, D>) -> Result<Self>
    where
        D: OneValue,
        B: FillOp<D>,
    {
        let layout = Layout::with_strides(
            other.layout.concrete_shape().clone(),
            other.layout.strides(),
            0,
        )?;
        let storage = other
            .backend
            .fill(&layout, tensor_creation::one_value::<D>())?;
        let tensor = Self::from_parts(other.backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn zeros_like(other: &Tensor<B, D>) -> Result<Self> {
        let layout = Layout::contiguous(other.layout.concrete_shape().clone());
        let numel = layout.num_elements();
        let storage = other.backend.allocator().allocate_zeroed(numel)?;
        let tensor = Self::from_parts(other.backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn full_like(other: &Tensor<B, D>, value: D) -> Result<Self>
    where
        B: FillOp<D>,
    {
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
        let len = tensor_creation::compute_arange_len(start, end, step)?;
        let shape = ConcreteShape::from_slice(&[len])?;
        let layout = Layout::contiguous(shape);
        let mut storage = backend.allocator().allocate(len)?;
        let values = tensor_creation::build_arange_values(len, start, step)?;
        backend.write(&mut storage, &layout, &values)?;
        let tensor = Self::from_parts(backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn eye(backend: &Arc<B>, rows: usize, cols: usize) -> Result<Self>
    where
        D: OneValue,
    {
        let shape = ConcreteShape::from_slice(&[rows, cols])?;
        let numel = shape.num_elements();
        let mut values = vec![D::default(); numel];
        let diag = rows.min(cols);
        if cols > 0 {
            let stride = cols;
            let diag_value = tensor_creation::one_value::<D>();
            for idx in 0..diag {
                values[idx * stride + idx] = diag_value;
            }
        }
        Self::from_vec(backend, values, &[rows, cols])
    }

    pub fn identity(backend: &Arc<B>, size: usize) -> Result<Self>
    where
        D: OneValue,
    {
        Self::eye(backend, size, size)
    }

    pub fn linspace(backend: &Arc<B>, start: D, end: D, steps: usize) -> Result<Self> {
        let values = tensor_creation::build_linspace_values(start, end, steps)?;
        let shape = ConcreteShape::from_slice(&[values.len()])?;
        let layout = Layout::contiguous(shape);
        let mut storage = backend.allocator().allocate(values.len())?;
        backend.write(&mut storage, &layout, &values)?;
        let tensor = Self::from_parts(backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn logspace(backend: &Arc<B>, start: D, end: D, steps: usize, base: D) -> Result<Self> {
        let values = tensor_creation::build_logspace_values(start, end, steps, base)?;
        let shape = ConcreteShape::from_slice(&[values.len()])?;
        let layout = Layout::contiguous(shape);
        let mut storage = backend.allocator().allocate(values.len())?;
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

    pub fn rank(&self) -> usize {
        self.layout.shape().len()
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn numel(&self) -> usize {
        self.layout.num_elements()
    }

    pub fn into_vec(self) -> Result<Vec<D>>
    where
        B: CopyOp<D>,
    {
        if self.layout.is_contiguous() && self.layout.offset_bytes() == 0 {
            let mut values = vec![D::default(); self.numel()];
            self.backend
                .read(&self.storage, &self.layout, &mut values)?;
            return Ok(values);
        }
        let tensor = self.contiguous()?;
        tensor.into_vec()
    }

    pub fn to_vec(&self) -> Result<Vec<D>>
    where
        B: CopyOp<D>,
    {
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

    pub fn i<I: TensorIndex>(&self, index: I) -> Result<Self> {
        let indexers = index.to_indexers(self.shape())?;
        let layout = self.layout.perform_indexing(&indexers, D::DTYPE)?;
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

    /// Returns a broadcast view sharing the same storage as `self`.
    ///
    /// Broadcasting follows NumPy semantics: dimensions must be compatible (size 1 or matching).
    /// If the source tensor has lower rank than the target shape, it is reshaped by prepending
    /// dimensions of size 1.
    ///
    /// The result is a view with zero-copy: broadcasted dimensions have stride 0.
    /// Call `.contiguous()` explicitly if you need a contiguous copy for performance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bolt_core::{Tensor, Backend, Result};
    /// # use bolt_cpu::CpuBackend;
    /// # use std::sync::Arc;
    /// # fn example<B: Backend<f32>>(backend: &Arc<B>) -> Result<()> {
    /// let tensor = Tensor::from_slice(backend, &[1.0, 2.0], &[2])?;
    /// let broadcasted = tensor.broadcast_to(&[3, 2])?;
    /// assert_eq!(broadcasted.shape(), &[3, 2]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `Error::ShapeMismatch` if the shapes are incompatible for broadcasting.
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self> {
        let target_shape = ConcreteShape::from_slice(shape)?;
        let src_shape = self.layout.concrete_shape().as_slice();
        let src_rank = src_shape.len();
        let target_rank = target_shape.as_slice().len();

        let mut current = self.clone();

        if src_rank < target_rank {
            let mut reshaped = vec![1; target_rank - src_rank];
            reshaped.extend_from_slice(src_shape);
            current = current.reshape(&reshaped)?;
        }

        let layout = current.layout().broadcast_to(&target_shape)?;
        current.validate_layout_for_storage(&current.storage, &layout)?;

        Ok(Self::from_parts(
            current.backend.clone(),
            current.storage.clone(),
            layout,
        ))
    }

    pub fn transpose(&self, axis_a: usize, axis_b: usize) -> Result<Self> {
        let layout = self.layout.transpose(axis_a, axis_b)?;
        self.validate_layout_for_storage(&self.storage, &layout)?;
        Ok(self.with_layout(layout))
    }

    pub fn contiguous(&self) -> Result<Self>
    where
        B: CopyOp<D>,
    {
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

    pub fn add(&self, other: &Self) -> Result<Self>
    where
        B: AddOp<D>,
    {
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

    pub fn sub(&self, other: &Self) -> Result<Self>
    where
        B: SubOp<D>,
    {
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

    pub fn mul(&self, other: &Self) -> Result<Self>
    where
        B: MulOp<D>,
    {
        self.ensure_same_backend(other)?;
        let parts = self
            .backend
            .mul(&self.storage, &other.storage, &self.layout, &other.layout)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn matmul(&self, other: &Self) -> Result<Self>
    where
        B: MatmulOp<D>,
    {
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
        B: Backend<f32> + MeanOp<D, F32Storage = <B as Backend<f32>>::Storage>,
        D: ToF32,
    {
        let parts = MeanOp::<D>::mean_f32(self.backend.as_ref(), &self.storage, &self.layout)?;
        Ok(Tensor::<B, f32>::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn neg(&self) -> Result<Self>
    where
        B: NegOp<D>,
    {
        let parts = self.backend.neg(&self.layout, &self.storage)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn abs(&self) -> Result<Self>
    where
        B: AbsOp<D>,
    {
        let parts = self.backend.abs(&self.layout, &self.storage)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn exp(&self) -> Result<Self>
    where
        B: ExpOp<D>,
        D: FloatType,
    {
        let parts = self.backend.exp(&self.layout, &self.storage)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn log(&self) -> Result<Self>
    where
        B: LogOp<D>,
        D: FloatType,
    {
        let parts = self.backend.log(&self.layout, &self.storage)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn sqrt(&self) -> Result<Self>
    where
        B: SqrtOp<D>,
        D: FloatType,
    {
        let parts = self.backend.sqrt(&self.layout, &self.storage)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn sin(&self) -> Result<Self>
    where
        B: SinOp<D>,
        D: FloatType,
    {
        let parts = self.backend.sin(&self.layout, &self.storage)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn cos(&self) -> Result<Self>
    where
        B: CosOp<D>,
        D: FloatType,
    {
        let parts = self.backend.cos(&self.layout, &self.storage)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn tanh(&self) -> Result<Self>
    where
        B: TanhOp<D>,
        D: FloatType,
    {
        let parts = self.backend.tanh(&self.layout, &self.storage)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn relu(&self) -> Result<Self>
    where
        B: ReluOp<D>,
    {
        let parts = self.backend.relu(&self.layout, &self.storage)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn div(&self, other: &Self) -> Result<Self>
    where
        B: DivOp<D>,
    {
        self.ensure_same_backend(other)?;
        let parts = self
            .backend
            .div(&self.storage, &other.storage, &self.layout, &other.layout)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn pow(&self, other: &Self) -> Result<Self>
    where
        B: PowOp<D>,
        D: FloatType,
    {
        self.ensure_same_backend(other)?;
        let parts = self
            .backend
            .pow(&self.storage, &other.storage, &self.layout, &other.layout)?;
        Ok(Self::from_parts(
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

    pub fn from_parts(backend: Arc<B>, storage: B::Storage, layout: Layout) -> Self {
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
}
