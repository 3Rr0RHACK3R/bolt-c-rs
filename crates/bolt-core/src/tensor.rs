use std::{marker::PhantomData, sync::Arc};

use crate::{
    allocator::StorageAllocator,
    backend::{
        AbsOp, AddOp, ArgmaxOp, ArgminOp, Backend, BroadcastToOp, CopyOp, CosOp, DivOp, ExpOp,
        FillOp, LogOp, MatmulOp, MaxOp, MeanOp, MinOp, MulOp, NegOp, PowOp, ProdOp, RandomOp, ReluOp,
        ReshapeOp, SinOp, SqrtOp, SqueezeOp, SubOp, SumOp, TanhOp, TransposeOp, UnsqueezeOp,
    },
    dtype::{Float, NativeType},
    error::{Error, Result},
    index::TensorIndex,
    layout::Layout,
    shape::ConcreteShape,
    utils::tensor_creation,
};

#[derive(Clone)]
pub struct Tensor<B, D>
where
    B: Backend,
    D: NativeType,
{
    backend: Arc<B>,
    storage: B::Storage<D>,
    layout: Layout,
    _marker: PhantomData<D>,
}

impl<B, D> Tensor<B, D>
where
    B: Backend,
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
        let mut storage = backend.allocator::<D>().allocate(data.len())?;
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
        let mut storage = backend.allocator::<D>().allocate(data.len())?;
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
        let storage = backend.allocator::<D>().allocate_zeroed(numel)?;
        let tensor = Self::from_parts(backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn ones(backend: &Arc<B>, shape: &[usize]) -> Result<Self>
    where
        B: FillOp<D>,
    {
        Self::full(backend, shape, D::one())
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
        B: FillOp<D>,
    {
        let layout = Layout::with_strides(
            other.layout.concrete_shape().clone(),
            other.layout.strides(),
            0,
        )?;
        let storage = other.backend.fill(&layout, D::one())?;
        let tensor = Self::from_parts(other.backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn zeros_like(other: &Tensor<B, D>) -> Result<Self> {
        let layout = Layout::contiguous(other.layout.concrete_shape().clone());
        let numel = layout.num_elements();
        let storage = other.backend.allocator::<D>().allocate_zeroed(numel)?;
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

    pub fn uniform(
        backend: &Arc<B>,
        shape: &[usize],
        low: D,
        high: D,
        seed: Option<u64>,
    ) -> Result<Self>
    where
        B: RandomOp<D>,
    {
        let parts = backend.uniform(shape, low, high, seed)?;
        let tensor = Self::from_parts(backend.clone(), parts.storage, parts.layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn normal(
        backend: &Arc<B>,
        shape: &[usize],
        mean: D,
        std: D,
        seed: Option<u64>,
    ) -> Result<Self>
    where
        B: RandomOp<D>,
    {
        let parts = backend.normal(shape, mean, std, seed)?;
        let tensor = Self::from_parts(backend.clone(), parts.storage, parts.layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn arange(backend: &Arc<B>, start: D, end: D, step: D) -> Result<Self> {
        let len = tensor_creation::compute_arange_len(start, end, step)?;
        let shape = ConcreteShape::from_slice(&[len])?;
        let layout = Layout::contiguous(shape);
        let mut storage = backend.allocator::<D>().allocate(len)?;
        let values = tensor_creation::build_arange_values(len, start, step)?;
        backend.write(&mut storage, &layout, &values)?;
        let tensor = Self::from_parts(backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn eye(backend: &Arc<B>, rows: usize, cols: usize) -> Result<Self> {
        let shape = ConcreteShape::from_slice(&[rows, cols])?;
        let numel = shape.num_elements();
        let mut values = vec![D::default(); numel];
        let diag = rows.min(cols);
        if cols > 0 {
            let stride = cols;
            for idx in 0..diag {
                values[idx * stride + idx] = D::one();
            }
        }
        Self::from_vec(backend, values, &[rows, cols])
    }

    pub fn identity(backend: &Arc<B>, size: usize) -> Result<Self> {
        Self::eye(backend, size, size)
    }

    pub fn linspace(backend: &Arc<B>, start: D, end: D, steps: usize) -> Result<Self> {
        let values = tensor_creation::build_linspace_values(start, end, steps)?;
        let shape = ConcreteShape::from_slice(&[values.len()])?;
        let layout = Layout::contiguous(shape);
        let mut storage = backend.allocator::<D>().allocate(values.len())?;
        backend.write(&mut storage, &layout, &values)?;
        let tensor = Self::from_parts(backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn logspace(backend: &Arc<B>, start: D, end: D, steps: usize, base: D) -> Result<Self> {
        let values = tensor_creation::build_logspace_values(start, end, steps, base)?;
        let shape = ConcreteShape::from_slice(&[values.len()])?;
        let layout = Layout::contiguous(shape);
        let mut storage = backend.allocator::<D>().allocate(values.len())?;
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

    pub fn reshape(&self, shape: &[usize]) -> Result<Self>
    where
        B: ReshapeOp<D>,
    {
        if shape.iter().product::<usize>() != self.numel() {
            return Err(Error::SizeMismatch {
                expected: self.numel(),
                actual: shape.iter().product(),
            });
        }
        let parts = self.backend.reshape(&self.storage, &self.layout, shape)?;
        self.validate_layout_for_storage(&parts.storage, &parts.layout)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn squeeze(&self) -> Result<Self>
    where
        B: SqueezeOp<D>,
    {
        let parts = self.backend.squeeze_all(&self.storage, &self.layout)?;
        self.validate_layout_for_storage(&parts.storage, &parts.layout)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn squeeze_axis(&self, axis: isize) -> Result<Self>
    where
        B: SqueezeOp<D>,
    {
        let parts = self
            .backend
            .squeeze_axis(&self.storage, &self.layout, axis)?;
        self.validate_layout_for_storage(&parts.storage, &parts.layout)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn unsqueeze(&self, axis: isize) -> Result<Self>
    where
        B: UnsqueezeOp<D>,
    {
        let parts = self
            .backend
            .unsqueeze_axis(&self.storage, &self.layout, axis)?;
        self.validate_layout_for_storage(&parts.storage, &parts.layout)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn expand(&self, shape: &[isize]) -> Result<Self>
    where
        B: BroadcastToOp<D> + ReshapeOp<D>,
    {
        let target = self.infer_expand_shape(shape)?;
        self.broadcast_to(&target)
    }

    pub fn expand_as(&self, other: &Self) -> Result<Self>
    where
        B: BroadcastToOp<D> + ReshapeOp<D>,
    {
        let shape: Vec<isize> = other.shape().iter().map(|&d| d as isize).collect();
        self.expand(&shape)
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

    pub fn permute(&self, axes: &[isize]) -> Result<Self> {
        let layout = self.layout.permute(axes)?;
        self.validate_layout_for_storage(&self.storage, &layout)?;
        Ok(self.with_layout(layout))
    }

    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self>
    where
        B: BroadcastToOp<D> + ReshapeOp<D>,
    {
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

        let parts = current
            .backend
            .broadcast_to(&current.storage, current.layout(), shape)?;
        current.validate_layout_for_storage(&parts.storage, &parts.layout)?;

        Ok(Self::from_parts(
            current.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn transpose(&self, axis_a: isize, axis_b: isize) -> Result<Self>
    where
        B: TransposeOp<D>,
    {
        let parts = self
            .backend
            .transpose(&self.storage, &self.layout, axis_a, axis_b)?;
        self.validate_layout_for_storage(&parts.storage, &parts.layout)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
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

    pub fn mean(&self, axes: Option<&[isize]>, keepdims: bool) -> Result<Self>
    where
        B: MeanOp<D>,
        D: Float,
    {
        let parts = self
            .backend
            .mean(&self.layout, &self.storage, axes, keepdims)?;
        Ok(Self::from_parts(
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
        D: Float,
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
        D: Float,
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
        D: Float,
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
        D: Float,
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
        D: Float,
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
        D: Float,
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
        D: Float,
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

    pub fn sum(&self, axes: Option<&[isize]>, keepdims: bool) -> Result<Self>
    where
        B: SumOp<D>,
    {
        let parts = self
            .backend
            .sum(&self.layout, &self.storage, axes, keepdims)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn prod(&self, axes: Option<&[isize]>, keepdims: bool) -> Result<Self>
    where
        B: ProdOp<D>,
    {
        let parts = self
            .backend
            .prod(&self.layout, &self.storage, axes, keepdims)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn min(&self, axes: Option<&[isize]>, keepdims: bool) -> Result<Self>
    where
        B: MinOp<D>,
    {
        let parts = self
            .backend
            .min(&self.layout, &self.storage, axes, keepdims)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn max(&self, axes: Option<&[isize]>, keepdims: bool) -> Result<Self>
    where
        B: MaxOp<D>,
    {
        let parts = self
            .backend
            .max(&self.layout, &self.storage, axes, keepdims)?;
        Ok(Self::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn argmin(&self, axes: Option<&[isize]>, keepdims: bool) -> Result<Tensor<B, i32>>
    where
        B: ArgminOp<D>,
    {
        let parts = self
            .backend
            .argmin(&self.layout, &self.storage, axes, keepdims)?;
        Ok(Tensor::<B, i32>::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    pub fn argmax(&self, axes: Option<&[isize]>, keepdims: bool) -> Result<Tensor<B, i32>>
    where
        B: ArgmaxOp<D>,
    {
        let parts = self
            .backend
            .argmax(&self.layout, &self.storage, axes, keepdims)?;
        Ok(Tensor::<B, i32>::from_parts(
            self.backend.clone(),
            parts.storage,
            parts.layout,
        ))
    }

    fn infer_expand_shape(&self, requested: &[isize]) -> Result<Vec<usize>> {
        if requested.is_empty() {
            return Ok(self.shape().to_vec());
        }
        let src_shape = self.shape();
        let src_rank = src_shape.len();
        let dst_rank = requested.len();
        if dst_rank < src_rank {
            return Err(Error::invalid_shape(
                "expand target rank must be >= tensor rank",
            ));
        }
        let mut target = Vec::with_capacity(dst_rank);
        for &dim in requested {
            if dim == -1 {
                target.push(0);
            } else if dim <= 0 {
                return Err(Error::invalid_shape(
                    "expand dimensions must be positive or -1",
                ));
            } else {
                target.push(dim as usize);
            }
        }
        let mut src_idx = src_rank as isize - 1;
        for dst_idx in (0..dst_rank).rev() {
            let has_src = src_idx >= 0;
            let src_dim = if has_src {
                src_shape[src_idx as usize]
            } else {
                1
            };
            let desired = if requested[dst_idx] == -1 {
                src_dim
            } else {
                target[dst_idx]
            };
            target[dst_idx] = desired;
            if src_dim == desired || src_dim == 1 {
                if has_src {
                    src_idx -= 1;
                }
                continue;
            }
            return Err(Error::ShapeMismatch {
                lhs: src_shape.to_vec(),
                rhs: target,
            });
        }
        while src_idx >= 0 {
            if src_shape[src_idx as usize] != 1 {
                return Err(Error::ShapeMismatch {
                    lhs: src_shape.to_vec(),
                    rhs: target.clone(),
                });
            }
            src_idx -= 1;
        }
        Ok(target)
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

    fn validate_layout_for_storage(&self, storage: &B::Storage<D>, layout: &Layout) -> Result<()> {
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

    pub fn from_parts(backend: Arc<B>, storage: B::Storage<D>, layout: Layout) -> Self {
        Self {
            backend,
            storage,
            layout,
            _marker: PhantomData,
        }
    }

    pub fn storage(&self) -> &B::Storage<D> {
        &self.storage
    }
}
