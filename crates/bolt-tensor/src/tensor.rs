use std::{marker::PhantomData, sync::Arc};

use bolt_core::{
    allocator::StorageAllocator,
    backend::{
        AbsOp, AddOp, ArgmaxOp, ArgminOp, Backend, BernoulliMaskOp, BroadcastToOp, CastOp, CopyOp,
        CosOp, DivOp, ExpOp, FillOp, LogOp, MatmulOp, MaxOp, MeanOp, MinOp, MulOp, NegOp, PowOp,
        ProdOp, RandomOp, ReluOp, ReshapeOp, SigmoidOp, SinOp, SqrtOp, SqueezeOp, SubOp, SumOp,
        TanhOp, TransposeOp, UnsqueezeOp,
    },
    dtype::{Float, NativeType},
    error::{Error, Result},
    index::TensorIndex,
    layout::Layout,
    shape::Shape,
};

use crate::{autograd, utils::tensor_creation};

pub trait ToBackend<B: Backend> {
    type Output;
    fn to_backend(self, backend: &Arc<B>) -> Result<Self::Output>;
}

impl<B1, B2, D> ToBackend<B2> for Tensor<B1, D>
where
    B1: Backend + CopyOp<D>,
    B2: Backend,
    D: NativeType,
{
    type Output = Tensor<B2, D>;

    fn to_backend(self, backend: &Arc<B2>) -> Result<Self::Output> {
        // Read tensor data to host memory (needs a better approach maybe?)
        let data = self.to_vec()?;
        let shape = self.shape().as_slice().to_vec();
        
        Tensor::<B2, D>::from_vec(backend, data, &shape)
    }
}

#[derive(Clone)]
pub struct Tensor<B, D>
where
    B: Backend,
    D: NativeType,
{
    backend: Arc<B>,
    storage: B::Storage<D>,
    layout: Layout,
    autograd: AutogradMeta,
    _marker: PhantomData<D>,
}

#[derive(Clone, Debug)]
struct AutogradMeta {
    origin: autograd::TensorId,
    requires_grad: bool,
}

impl AutogradMeta {
    fn new() -> Self {
        Self {
            origin: autograd::next_tensor_id(),
            requires_grad: false,
        }
    }
}

impl<B, D> Tensor<B, D>
where
    B: Backend,
    D: NativeType,
{
    pub fn detach(&self) -> Self {
        self.view_from_parts(self.storage.clone(), self.layout.clone())
    }

    pub fn cast<Dst>(&self) -> Result<Tensor<B, Dst>>
    where
        B: CastOp<D, Dst> + 'static,
        Dst: NativeType,
    {
        if autograd::grad_enabled() && self.requires_grad_enabled() {
            return Err(Error::OpError(
                "cast is not differentiable yet; call detach() or use autograd::no_grad()".into(),
            ));
        }

        let parts = self.backend.cast(&self.storage, &self.layout)?;
        let tensor =
            Tensor::<B, Dst>::from_parts(self.backend.clone(), parts.storage, parts.layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn with_requires_grad(mut self, requires_grad: bool) -> Self
    where
        D: Float,
    {
        self.autograd.requires_grad = requires_grad;
        self
    }

    pub fn requires_grad(self) -> Self
    where
        D: Float,
    {
        self.with_requires_grad(true)
    }

    pub fn backward(&self) -> Result<autograd::Grads<B, D>>
    where
        B: AddOp<D> + FillOp<D> + 'static,
        D: Float + 'static,
    {
        autograd::backward(self)
    }

    pub(crate) fn requires_grad_enabled(&self) -> bool {
        self.autograd.requires_grad
    }

    pub(crate) fn tensor_id(&self) -> autograd::TensorId {
        self.autograd.origin
    }

    pub fn from_slice(backend: &Arc<B>, data: &[D], shape: &[usize]) -> Result<Self> {
        let shape = Shape::from_slice(shape)?;
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
        let shape = Shape::from_slice(shape)?;
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
        let shape = Shape::from_slice(shape)?;
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
        let shape = Shape::from_slice(shape)?;
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
        let layout = Layout::with_strides(other.layout.shape().clone(), other.layout.strides(), 0)?;
        let storage = other.backend.fill(&layout, D::one())?;
        let tensor = Self::from_parts(other.backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn zeros_like(other: &Tensor<B, D>) -> Result<Self> {
        let layout = Layout::contiguous(other.layout.shape().clone());
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
        let layout = Layout::with_strides(other.layout.shape().clone(), other.layout.strides(), 0)?;
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

    pub fn bernoulli_mask(
        backend: &Arc<B>,
        shape: &[usize],
        p_keep: D,
        seed: Option<u64>,
    ) -> Result<Self>
    where
        B: BernoulliMaskOp<D>,
        D: Float,
    {
        let parts = backend.bernoulli_mask(shape, p_keep, seed)?;
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
        let shape = Shape::from_slice(&[len])?;
        let layout = Layout::contiguous(shape);
        let mut storage = backend.allocator::<D>().allocate(len)?;
        let values = tensor_creation::build_arange_values(len, start, step)?;
        backend.write(&mut storage, &layout, &values)?;
        let tensor = Self::from_parts(backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn eye(backend: &Arc<B>, rows: usize, cols: usize) -> Result<Self> {
        let shape = Shape::from_slice(&[rows, cols])?;
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

    pub fn one_hot(indices: &Tensor<B, i32>, num_classes: usize) -> Result<Self>
    where
        B: CopyOp<i32>,
    {
        let indices_vec = indices.to_vec()?;
        let n = indices_vec.len();
        let numel = n * num_classes;
        let mut values = vec![D::default(); numel];

        for (i, &idx) in indices_vec.iter().enumerate() {
            if idx < 0 || (idx as usize) >= num_classes {
                return Err(Error::invalid_shape(
                    "one_hot index out of bounds for num_classes",
                ));
            }
            values[i * num_classes + idx as usize] = D::one();
        }

        Tensor::<B, D>::from_vec(&indices.backend(), values, &[n, num_classes])
    }

    pub fn linspace(backend: &Arc<B>, start: D, end: D, steps: usize) -> Result<Self> {
        let values = tensor_creation::build_linspace_values(start, end, steps)?;
        let shape = Shape::from_slice(&[values.len()])?;
        let layout = Layout::contiguous(shape);
        let mut storage = backend.allocator::<D>().allocate(values.len())?;
        backend.write(&mut storage, &layout, &values)?;
        let tensor = Self::from_parts(backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn logspace(backend: &Arc<B>, start: D, end: D, steps: usize, base: D) -> Result<Self> {
        let values = tensor_creation::build_logspace_values(start, end, steps, base)?;
        let shape = Shape::from_slice(&[values.len()])?;
        let layout = Layout::contiguous(shape);
        let mut storage = backend.allocator::<D>().allocate(values.len())?;
        backend.write(&mut storage, &layout, &values)?;
        let tensor = Self::from_parts(backend.clone(), storage, layout);
        tensor.validate_layout_for_storage(&tensor.storage, &tensor.layout)?;
        Ok(tensor)
    }

    pub fn stack<'a, I>(backend: &Arc<B>, tensors: I) -> Result<Self>
    where
        I: IntoIterator<Item = &'a Tensor<B, D>>,
        B: CopyOp<D>,
        D: 'a,
    {
        let tensors: Vec<_> = tensors.into_iter().collect();
        if tensors.is_empty() {
            return Err(Error::invalid_shape("cannot stack empty tensor list"));
        }
        let first_shape = tensors[0].shape();
        for t in &tensors[1..] {
            if t.shape() != first_shape {
                return Err(Error::invalid_shape(
                    "all tensors must have the same shape for stacking",
                ));
            }
        }
        let n = tensors.len();
        let elem_count = first_shape.iter().product::<usize>();
        let total = n * elem_count;
        let mut buf = Vec::with_capacity(total);
        for t in &tensors {
            buf.extend_from_slice(&t.to_vec()?);
        }
        let mut out_shape = Vec::with_capacity(first_shape.len() + 1);
        out_shape.push(n);
        out_shape.extend_from_slice(first_shape.as_slice());
        Tensor::from_vec(backend, buf, &out_shape)
    }

    pub fn from_iter<I>(backend: &Arc<B>, iter: I) -> Result<Self>
    where
        I: IntoIterator<Item = D>,
    {
        let data: Vec<D> = iter.into_iter().collect();
        let len = data.len();
        if len == 0 {
            return Err(Error::invalid_shape(
                "cannot create tensor from empty iterator",
            ));
        }
        Tensor::from_vec(backend, data, &[len])
    }

    pub fn backend(&self) -> Arc<B> {
        self.backend.clone()
    }

    pub fn device(&self) -> &B::Device {
        self.backend.device()
    }

    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    pub fn rank(&self) -> usize {
        self.shape().len()
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
        let mut out = self.view_from_parts(parts.storage, parts.layout);
        self.record_reshape(&mut out)?;
        Ok(out)
    }

    pub fn squeeze(&self) -> Result<Self>
    where
        B: SqueezeOp<D> + UnsqueezeOp<D> + 'static,
    {
        let parts = self.backend.squeeze_all(&self.storage, &self.layout)?;
        self.validate_layout_for_storage(&parts.storage, &parts.layout)?;
        let mut out = self.view_from_parts(parts.storage, parts.layout);
        self.record_squeeze_all(&mut out)?;
        Ok(out)
    }

    pub fn squeeze_axis(&self, axis: isize) -> Result<Self>
    where
        B: SqueezeOp<D> + UnsqueezeOp<D> + 'static,
    {
        let parts = self
            .backend
            .squeeze_axis(&self.storage, &self.layout, axis)?;
        self.validate_layout_for_storage(&parts.storage, &parts.layout)?;
        let mut out = self.view_from_parts(parts.storage, parts.layout);
        self.record_squeeze_axis(axis, &mut out)?;
        Ok(out)
    }

    pub fn unsqueeze(&self, axis: isize) -> Result<Self>
    where
        B: SqueezeOp<D> + UnsqueezeOp<D> + 'static,
    {
        let parts = self
            .backend
            .unsqueeze_axis(&self.storage, &self.layout, axis)?;
        self.validate_layout_for_storage(&parts.storage, &parts.layout)?;
        let mut out = self.view_from_parts(parts.storage, parts.layout);
        self.record_unsqueeze(axis, &mut out)?;
        Ok(out)
    }

    pub fn expand(&self, shape: &[isize]) -> Result<Self>
    where
        B: BroadcastToOp<D> + ReshapeOp<D> + SumOp<D> + 'static,
    {
        let target = self.infer_expand_shape(shape)?;
        self.broadcast_to(&target)
    }

    pub fn expand_as(&self, other: &Self) -> Result<Self>
    where
        B: BroadcastToOp<D> + ReshapeOp<D> + SumOp<D> + 'static,
    {
        let shape: Vec<isize> = other.shape().iter().map(|&d| d as isize).collect();
        self.expand(&shape)
    }

    pub fn i<I: TensorIndex>(&self, index: I) -> Result<Self> {
        let indexers = index.to_indexers(self.shape().as_slice())?;
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
        B: BroadcastToOp<D> + ReshapeOp<D> + SumOp<D> + 'static,
    {
        let target_shape = Shape::from_slice(shape)?;
        let src_shape = self.layout.shape().as_slice();
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

        let mut out = current.view_from_parts(parts.storage, parts.layout);
        current.record_broadcast_to(shape, &mut out)?;
        Ok(out)
    }

    pub fn transpose(&self, axis_a: isize, axis_b: isize) -> Result<Self>
    where
        B: TransposeOp<D>,
    {
        let parts = self
            .backend
            .transpose(&self.storage, &self.layout, axis_a, axis_b)?;
        self.validate_layout_for_storage(&parts.storage, &parts.layout)?;
        let mut out = self.view_from_parts(parts.storage, parts.layout);
        self.record_transpose(axis_a, axis_b, &mut out)?;
        Ok(out)
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
        Ok(self.view_from_parts(parts.storage, parts.layout))
    }

    pub fn add(&self, other: &Self) -> Result<Self>
    where
        B: AddOp<D> + ReshapeOp<D> + SumOp<D> + 'static,
    {
        self.ensure_same_backend(other)?;
        let parts = self
            .backend
            .add(&self.storage, &other.storage, &self.layout, &other.layout)?;
        let mut out = Self::from_parts(self.backend.clone(), parts.storage, parts.layout);
        self.record_add(other, &mut out)?;
        Ok(out)
    }

    pub fn sub(&self, other: &Self) -> Result<Self>
    where
        B: CopyOp<D> + NegOp<D> + ReshapeOp<D> + SubOp<D> + SumOp<D> + 'static,
        D: std::ops::Neg<Output = D> + std::ops::Sub<Output = D> + 'static,
    {
        self.ensure_same_backend(other)?;
        let parts = self
            .backend
            .sub(&self.storage, &other.storage, &self.layout, &other.layout)?;
        let mut out = Self::from_parts(self.backend.clone(), parts.storage, parts.layout);
        self.record_sub(other, &mut out)?;
        Ok(out)
    }

    pub fn mul(&self, other: &Self) -> Result<Self>
    where
        B: CopyOp<D> + MulOp<D> + ReshapeOp<D> + SumOp<D> + 'static,
        D: std::ops::Mul<Output = D> + 'static,
    {
        self.ensure_same_backend(other)?;
        let parts = self
            .backend
            .mul(&self.storage, &other.storage, &self.layout, &other.layout)?;
        let mut out = Self::from_parts(self.backend.clone(), parts.storage, parts.layout);
        self.record_mul(other, &mut out)?;
        Ok(out)
    }

    pub fn matmul(&self, other: &Self) -> Result<Self>
    where
        B: MatmulOp<D> + TransposeOp<D> + 'static,
    {
        self.ensure_same_backend(other)?;
        let parts =
            self.backend
                .matmul(&self.storage, &other.storage, &self.layout, &other.layout)?;
        let mut out = Self::from_parts(self.backend.clone(), parts.storage, parts.layout);
        self.record_matmul(other, &mut out)?;
        Ok(out)
    }

    pub fn mean(&self, axes: Option<&[isize]>, keepdims: bool) -> Result<Self>
    where
        B: CopyOp<D> + MeanOp<D> + 'static,
        D: Float + 'static,
    {
        let parts = self
            .backend
            .mean(&self.layout, &self.storage, axes, keepdims)?;
        let mut out = Self::from_parts(self.backend.clone(), parts.storage, parts.layout);
        self.record_mean(axes, keepdims, &mut out)?;
        Ok(out)
    }

    pub fn neg(&self) -> Result<Self>
    where
        B: CopyOp<D> + NegOp<D> + 'static,
        D: std::ops::Neg<Output = D> + 'static,
    {
        let parts = self.backend.neg(&self.layout, &self.storage)?;
        let mut out = Self::from_parts(self.backend.clone(), parts.storage, parts.layout);
        self.record_neg(&mut out)?;
        Ok(out)
    }

    pub fn abs(&self) -> Result<Self>
    where
        B: CopyOp<D> + AbsOp<D> + 'static,
        D: PartialOrd + std::ops::Neg<Output = D> + std::ops::Mul<Output = D> + 'static,
    {
        let parts = self.backend.abs(&self.layout, &self.storage)?;
        let mut out = Self::from_parts(self.backend.clone(), parts.storage, parts.layout);
        self.record_abs(&mut out)?;
        Ok(out)
    }

    pub fn exp(&self) -> Result<Self>
    where
        B: CopyOp<D> + ExpOp<D> + 'static,
        D: Float + 'static,
    {
        let parts = self.backend.exp(&self.layout, &self.storage)?;
        let mut out = Self::from_parts(self.backend.clone(), parts.storage, parts.layout);
        self.record_exp(&mut out)?;
        Ok(out)
    }

    pub fn log(&self) -> Result<Self>
    where
        B: CopyOp<D> + LogOp<D> + 'static,
        D: Float + 'static,
    {
        let parts = self.backend.log(&self.layout, &self.storage)?;
        let mut out = Self::from_parts(self.backend.clone(), parts.storage, parts.layout);
        self.record_log(&mut out)?;
        Ok(out)
    }

    pub fn sqrt(&self) -> Result<Self>
    where
        B: CopyOp<D> + SqrtOp<D> + 'static,
        D: Float + 'static,
    {
        let parts = self.backend.sqrt(&self.layout, &self.storage)?;
        let mut out = Self::from_parts(self.backend.clone(), parts.storage, parts.layout);
        self.record_sqrt(&mut out)?;
        Ok(out)
    }

    pub fn sin(&self) -> Result<Self>
    where
        B: CopyOp<D> + SinOp<D> + 'static,
        D: Float + 'static,
    {
        let parts = self.backend.sin(&self.layout, &self.storage)?;
        let mut out = Self::from_parts(self.backend.clone(), parts.storage, parts.layout);
        self.record_sin(&mut out)?;
        Ok(out)
    }

    pub fn cos(&self) -> Result<Self>
    where
        B: CopyOp<D> + CosOp<D> + 'static,
        D: Float + 'static,
    {
        let parts = self.backend.cos(&self.layout, &self.storage)?;
        let mut out = Self::from_parts(self.backend.clone(), parts.storage, parts.layout);
        self.record_cos(&mut out)?;
        Ok(out)
    }

    pub fn tanh(&self) -> Result<Self>
    where
        B: CopyOp<D> + TanhOp<D> + 'static,
        D: Float + 'static,
    {
        let parts = self.backend.tanh(&self.layout, &self.storage)?;
        let mut out = Self::from_parts(self.backend.clone(), parts.storage, parts.layout);
        self.record_tanh(&mut out)?;
        Ok(out)
    }

    pub fn relu(&self) -> Result<Self>
    where
        B: CopyOp<D> + ReluOp<D> + 'static,
        D: PartialOrd + std::ops::Mul<Output = D> + 'static,
    {
        let parts = self.backend.relu(&self.layout, &self.storage)?;
        let mut out = Self::from_parts(self.backend.clone(), parts.storage, parts.layout);
        self.record_relu(&mut out)?;
        Ok(out)
    }

    pub fn sigmoid(&self) -> Result<Self>
    where
        B: CopyOp<D> + SigmoidOp<D> + 'static,
        D: Float + std::ops::Mul<Output = D> + std::ops::Sub<Output = D> + 'static,
    {
        let parts = self.backend.sigmoid(&self.layout, &self.storage)?;
        let mut out = Self::from_parts(self.backend.clone(), parts.storage, parts.layout);
        self.record_sigmoid(&mut out)?;
        Ok(out)
    }

    pub fn softmax(&self, dim: Option<isize>) -> Result<Self>
    where
        B: CopyOp<D>
            + ExpOp<D>
            + LogOp<D>
            + MaxOp<D>
            + SubOp<D>
            + SumOp<D>
            + ReshapeOp<D>
            + NegOp<D>
            + 'static,
        D: Float + PartialOrd + PartialEq + 'static,
    {
        let dim = dim.unwrap_or(-1);
        let axes = &[dim];
        let max = self.max(Some(axes), true)?;
        let shifted = self.sub(&max)?;
        let logsumexp = shifted.exp()?.sum(Some(axes), true)?.log()?;
        let log_softmax = shifted.sub(&logsumexp)?;
        log_softmax.exp()
    }

    // TODO: Why the need for std ops trait bound here?
    pub fn div(&self, other: &Self) -> Result<Self>
    where
        B: CopyOp<D> + DivOp<D> + MulOp<D> + NegOp<D> + ReshapeOp<D> + SumOp<D> + 'static,
        D: std::ops::Div<Output = D>
            + std::ops::Mul<Output = D>
            + std::ops::Neg<Output = D>
            + 'static,
    {
        self.ensure_same_backend(other)?;
        let parts = self
            .backend
            .div(&self.storage, &other.storage, &self.layout, &other.layout)?;
        let mut out = Self::from_parts(self.backend.clone(), parts.storage, parts.layout);
        self.record_div(other, &mut out)?;
        Ok(out)
    }

    pub fn pow(&self, other: &Self) -> Result<Self>
    where
        B: CopyOp<D> + FillOp<D> + LogOp<D> + MulOp<D> + NegOp<D> + PowOp<D> + SubOp<D> + 'static,
        D: Float + 'static,
    {
        self.ensure_same_backend(other)?;
        if autograd::grad_enabled()
            && (self.autograd.requires_grad || other.autograd.requires_grad)
            && self.shape() != other.shape()
        {
            return Err(Error::OpError(
                "autograd pow does not support broadcasting yet".into(),
            ));
        }
        let parts = self
            .backend
            .pow(&self.storage, &other.storage, &self.layout, &other.layout)?;
        let mut out = Self::from_parts(self.backend.clone(), parts.storage, parts.layout);
        self.record_pow(other, &mut out)?;
        Ok(out)
    }

    pub fn sum(&self, axes: Option<&[isize]>, keepdims: bool) -> Result<Self>
    where
        B: CopyOp<D> + SumOp<D> + 'static,
    {
        let parts = self
            .backend
            .sum(&self.layout, &self.storage, axes, keepdims)?;
        let mut out = Self::from_parts(self.backend.clone(), parts.storage, parts.layout);
        self.record_sum(axes, keepdims, &mut out)?;
        Ok(out)
    }

    pub fn prod(&self, axes: Option<&[isize]>, keepdims: bool) -> Result<Self>
    where
        B: CopyOp<D> + ProdOp<D> + 'static,
        D: PartialEq + std::ops::Mul<Output = D> + std::ops::Div<Output = D> + 'static,
    {
        let parts = self
            .backend
            .prod(&self.layout, &self.storage, axes, keepdims)?;
        let mut out = Self::from_parts(self.backend.clone(), parts.storage, parts.layout);
        self.record_prod(axes, keepdims, &mut out)?;
        Ok(out)
    }

    pub fn min(&self, axes: Option<&[isize]>, keepdims: bool) -> Result<Self>
    where
        B: CopyOp<D> + MinOp<D> + 'static,
        D: PartialOrd + PartialEq + std::ops::Div<Output = D> + 'static,
    {
        let parts = self
            .backend
            .min(&self.layout, &self.storage, axes, keepdims)?;
        let mut out = Self::from_parts(self.backend.clone(), parts.storage, parts.layout);
        self.record_min(axes, keepdims, &mut out)?;
        Ok(out)
    }

    pub fn max(&self, axes: Option<&[isize]>, keepdims: bool) -> Result<Self>
    where
        B: CopyOp<D> + MaxOp<D> + 'static,
        D: PartialOrd + PartialEq + std::ops::Div<Output = D> + 'static,
    {
        let parts = self
            .backend
            .max(&self.layout, &self.storage, axes, keepdims)?;
        let mut out = Self::from_parts(self.backend.clone(), parts.storage, parts.layout);
        self.record_max(axes, keepdims, &mut out)?;
        Ok(out)
    }

    pub fn argmin(&self, axes: Option<&[isize]>, keepdims: bool) -> Result<Tensor<B, i32>>
    where
        B: ArgminOp<D>,
    {
        self.ensure_nondifferentiable_op("argmin")?;
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
        self.ensure_nondifferentiable_op("argmax")?;
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
        let self_device_id = self.backend.device_id();
        let other_device_id = other.backend.device_id();

        if self_device_id != other_device_id {
            return Err(Error::Device(format!(
                "tensors are on different devices: {:?} vs {:?}",
                self_device_id, other_device_id
            )));
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
            autograd: AutogradMeta::new(),
            _marker: PhantomData,
        }
    }

    pub fn from_parts(backend: Arc<B>, storage: B::Storage<D>, layout: Layout) -> Self {
        Self {
            backend,
            storage,
            layout,
            autograd: AutogradMeta::new(),
            _marker: PhantomData,
        }
    }

    pub fn storage(&self) -> &B::Storage<D> {
        &self.storage
    }

    fn view_from_parts(&self, storage: B::Storage<D>, layout: Layout) -> Self {
        Self {
            backend: self.backend.clone(),
            storage,
            layout,
            autograd: AutogradMeta::new(),
            _marker: PhantomData,
        }
    }

    fn record_binary_node(
        &self,
        other: &Self,
        out: &mut Self,
        op: Box<dyn autograd::BackwardOp<B, D>>,
        saved_tensors: Vec<Self>,
    ) where
        B: 'static,
        D: NativeType + 'static,
    {
        let requires_grad = D::SUPPORTS_GRAD
            && autograd::grad_enabled()
            && (self.autograd.requires_grad || other.autograd.requires_grad);
        out.autograd.requires_grad = requires_grad;

        if !requires_grad {
            return;
        }

        autograd::record_node::<B, D>(
            out.autograd.origin,
            vec![
                (self.autograd.origin, self.autograd.requires_grad),
                (other.autograd.origin, other.autograd.requires_grad),
            ],
            op,
            saved_tensors,
        );
    }

    fn record_unary_node(
        &self,
        out: &mut Self,
        op: Box<dyn autograd::BackwardOp<B, D>>,
        saved_tensors: Vec<Self>,
    ) where
        B: 'static,
        D: NativeType + 'static,
    {
        let requires_grad =
            D::SUPPORTS_GRAD && autograd::grad_enabled() && self.autograd.requires_grad;
        out.autograd.requires_grad = requires_grad;

        if !requires_grad {
            return;
        }

        autograd::record_node::<B, D>(
            out.autograd.origin,
            vec![(self.autograd.origin, self.autograd.requires_grad)],
            op,
            saved_tensors,
        );
    }

    fn record_reshape(&self, out: &mut Self) -> Result<()>
    where
        B: ReshapeOp<D> + 'static,
    {
        let op = Box::new(autograd::ReshapeBackward::new(self.shape().to_vec()));
        self.record_unary_node(out, op, vec![]);
        Ok(())
    }

    fn record_transpose(&self, axis_a: isize, axis_b: isize, out: &mut Self) -> Result<()>
    where
        B: TransposeOp<D> + 'static,
    {
        let op = Box::new(autograd::TransposeBackward::new(axis_a, axis_b));
        self.record_unary_node(out, op, vec![]);
        Ok(())
    }

    fn record_unsqueeze(&self, axis: isize, out: &mut Self) -> Result<()>
    where
        B: SqueezeOp<D> + UnsqueezeOp<D> + 'static,
    {
        let op = Box::new(autograd::UnsqueezeBackward::new(axis));
        self.record_unary_node(out, op, vec![]);
        Ok(())
    }

    fn record_squeeze_axis(&self, axis: isize, out: &mut Self) -> Result<()>
    where
        B: SqueezeOp<D> + UnsqueezeOp<D> + 'static,
    {
        let op = Box::new(autograd::SqueezeAxisBackward::new(axis));
        self.record_unary_node(out, op, vec![]);
        Ok(())
    }

    fn record_squeeze_all(&self, out: &mut Self) -> Result<()>
    where
        B: SqueezeOp<D> + UnsqueezeOp<D> + 'static,
    {
        let axes: Vec<usize> = self
            .shape()
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| (d == 1).then_some(i))
            .collect();
        let op = Box::new(autograd::SqueezeAllBackward::new(axes));
        self.record_unary_node(out, op, vec![]);
        Ok(())
    }

    fn record_broadcast_to(&self, output_shape: &[usize], out: &mut Self) -> Result<()>
    where
        B: ReshapeOp<D> + SumOp<D> + 'static,
    {
        let input_shape = self.shape().to_vec();
        let op = Box::new(autograd::BroadcastToBackward::new(
            input_shape,
            output_shape.to_vec(),
        ));
        self.record_unary_node(out, op, vec![]);
        Ok(())
    }

    fn record_add(&self, other: &Self, out: &mut Self) -> Result<()>
    where
        B: ReshapeOp<D> + SumOp<D> + 'static,
        D: NativeType + 'static,
    {
        let op = Box::new(autograd::AddBackward::new(
            self.shape().to_vec(),
            other.shape().to_vec(),
        ));
        self.record_binary_node(other, out, op, vec![]);
        Ok(())
    }

    fn record_sub(&self, other: &Self, out: &mut Self) -> Result<()>
    where
        B: CopyOp<D> + NegOp<D> + ReshapeOp<D> + SumOp<D> + 'static,
        D: NativeType + std::ops::Neg<Output = D> + 'static,
    {
        let op = Box::new(autograd::SubBackward::new(
            self.shape().to_vec(),
            other.shape().to_vec(),
        ));
        self.record_binary_node(other, out, op, vec![]);
        Ok(())
    }

    fn record_mul(&self, other: &Self, out: &mut Self) -> Result<()>
    where
        B: CopyOp<D> + MulOp<D> + ReshapeOp<D> + SumOp<D> + 'static,
        D: NativeType + std::ops::Mul<Output = D> + 'static,
    {
        let op = Box::new(autograd::MulBackward::new(
            self.shape().to_vec(),
            other.shape().to_vec(),
        ));
        self.record_binary_node(other, out, op, vec![self.clone(), other.clone()]);
        Ok(())
    }

    fn record_matmul(&self, other: &Self, out: &mut Self) -> Result<()>
    where
        B: MatmulOp<D> + TransposeOp<D> + 'static,
        D: NativeType + 'static,
    {
        let op = Box::new(autograd::MatmulBackward::new());
        self.record_binary_node(other, out, op, vec![self.clone(), other.clone()]);
        Ok(())
    }

    fn record_sum(&self, axes: Option<&[isize]>, keepdims: bool, out: &mut Self) -> Result<()>
    where
        B: CopyOp<D> + 'static,
    {
        let requires_grad =
            D::SUPPORTS_GRAD && autograd::grad_enabled() && self.autograd.requires_grad;
        out.autograd.requires_grad = requires_grad;

        if !requires_grad {
            return Ok(());
        }

        let axes = autograd::canonical_axes(axes, self.rank())?;
        let op = Box::new(autograd::SumBackward::new(
            self.shape().to_vec(),
            axes,
            keepdims,
        ));
        autograd::record_node::<B, D>(
            out.autograd.origin,
            vec![(self.autograd.origin, self.autograd.requires_grad)],
            op,
            vec![],
        );

        Ok(())
    }

    fn ensure_nondifferentiable_op(&self, name: &str) -> Result<()> {
        if autograd::grad_enabled() && self.requires_grad_enabled() {
            return Err(Error::OpError(
                format!("{name} is not differentiable; call detach() or use autograd::no_grad()")
                    .into(),
            ));
        }
        Ok(())
    }

    fn record_mean(&self, axes: Option<&[isize]>, keepdims: bool, out: &mut Self) -> Result<()>
    where
        B: CopyOp<D> + 'static,
        D: Float + 'static,
    {
        let axes = autograd::canonical_axes(axes, self.rank())?;
        let op = Box::new(autograd::MeanBackward::new(
            self.shape().to_vec(),
            axes,
            keepdims,
        ));
        self.record_unary_node(out, op, vec![]);
        Ok(())
    }

    fn record_prod(&self, axes: Option<&[isize]>, keepdims: bool, out: &mut Self) -> Result<()>
    where
        B: CopyOp<D> + 'static,
        D: NativeType + PartialEq + std::ops::Mul<Output = D> + std::ops::Div<Output = D> + 'static,
    {
        let axes = autograd::canonical_axes(axes, self.rank())?;
        let op = Box::new(autograd::ProdBackward::new(
            self.shape().to_vec(),
            axes,
            keepdims,
        ));
        self.record_unary_node(out, op, vec![self.clone()]);
        Ok(())
    }

    fn record_min(&self, axes: Option<&[isize]>, keepdims: bool, out: &mut Self) -> Result<()>
    where
        B: CopyOp<D> + 'static,
        D: NativeType + PartialOrd + PartialEq + std::ops::Div<Output = D> + 'static,
    {
        let axes = autograd::canonical_axes(axes, self.rank())?;
        let op = Box::new(autograd::MinBackward::new(
            self.shape().to_vec(),
            axes,
            keepdims,
        ));
        self.record_unary_node(out, op, vec![self.clone()]);
        Ok(())
    }

    fn record_max(&self, axes: Option<&[isize]>, keepdims: bool, out: &mut Self) -> Result<()>
    where
        B: CopyOp<D> + 'static,
        D: NativeType + PartialOrd + PartialEq + std::ops::Div<Output = D> + 'static,
    {
        let axes = autograd::canonical_axes(axes, self.rank())?;
        let op = Box::new(autograd::MaxBackward::new(
            self.shape().to_vec(),
            axes,
            keepdims,
        ));
        self.record_unary_node(out, op, vec![self.clone()]);
        Ok(())
    }

    fn record_neg(&self, out: &mut Self) -> Result<()>
    where
        B: CopyOp<D> + NegOp<D> + 'static,
        D: NativeType + std::ops::Neg<Output = D> + 'static,
    {
        let op = Box::new(autograd::NegBackward::new());
        self.record_unary_node(out, op, vec![]);
        Ok(())
    }

    fn record_abs(&self, out: &mut Self) -> Result<()>
    where
        B: CopyOp<D> + 'static,
        D: NativeType
            + PartialOrd
            + std::ops::Neg<Output = D>
            + std::ops::Mul<Output = D>
            + 'static,
    {
        let op = Box::new(autograd::AbsBackward::new());
        self.record_unary_node(out, op, vec![self.clone()]);
        Ok(())
    }

    fn record_exp(&self, out: &mut Self) -> Result<()>
    where
        B: CopyOp<D> + 'static,
        D: Float + 'static,
    {
        let op = Box::new(autograd::ExpBackward::new());
        self.record_unary_node(out, op, vec![out.clone()]);
        Ok(())
    }

    fn record_log(&self, out: &mut Self) -> Result<()>
    where
        B: CopyOp<D> + 'static,
        D: Float + 'static,
    {
        let op = Box::new(autograd::LogBackward::new());
        self.record_unary_node(out, op, vec![self.clone()]);
        Ok(())
    }

    fn record_sqrt(&self, out: &mut Self) -> Result<()>
    where
        B: CopyOp<D> + 'static,
        D: Float + 'static,
    {
        let op = Box::new(autograd::SqrtBackward::new());
        self.record_unary_node(out, op, vec![out.clone()]);
        Ok(())
    }

    fn record_sin(&self, out: &mut Self) -> Result<()>
    where
        B: CopyOp<D> + 'static,
        D: Float + 'static,
    {
        let op = Box::new(autograd::SinBackward::new());
        self.record_unary_node(out, op, vec![self.clone()]);
        Ok(())
    }

    fn record_cos(&self, out: &mut Self) -> Result<()>
    where
        B: CopyOp<D> + 'static,
        D: Float + 'static,
    {
        let op = Box::new(autograd::CosBackward::new());
        self.record_unary_node(out, op, vec![self.clone()]);
        Ok(())
    }

    fn record_tanh(&self, out: &mut Self) -> Result<()>
    where
        B: CopyOp<D> + 'static,
        D: Float + 'static,
    {
        let op = Box::new(autograd::TanhBackward::new());
        self.record_unary_node(out, op, vec![out.clone()]);
        Ok(())
    }

    fn record_relu(&self, out: &mut Self) -> Result<()>
    where
        B: CopyOp<D> + 'static,
        D: NativeType + PartialOrd + std::ops::Mul<Output = D> + 'static,
    {
        let op = Box::new(autograd::ReluBackward::new());
        self.record_unary_node(out, op, vec![self.clone()]);
        Ok(())
    }

    fn record_sigmoid(&self, out: &mut Self) -> Result<()>
    where
        B: CopyOp<D> + 'static,
        D: Float + std::ops::Mul<Output = D> + std::ops::Sub<Output = D> + 'static,
    {
        let op = Box::new(autograd::SigmoidBackward::new());
        self.record_unary_node(out, op, vec![out.clone()]);
        Ok(())
    }

    fn record_div(&self, other: &Self, out: &mut Self) -> Result<()>
    where
        B: CopyOp<D> + DivOp<D> + MulOp<D> + NegOp<D> + ReshapeOp<D> + SumOp<D> + 'static,
        D: NativeType
            + std::ops::Div<Output = D>
            + std::ops::Mul<Output = D>
            + std::ops::Neg<Output = D>
            + 'static,
    {
        let op = Box::new(autograd::DivBackward::new(
            self.shape().to_vec(),
            other.shape().to_vec(),
        ));
        self.record_binary_node(other, out, op, vec![self.clone(), other.clone()]);
        Ok(())
    }

    fn record_pow(&self, other: &Self, out: &mut Self) -> Result<()>
    where
        B: CopyOp<D> + FillOp<D> + LogOp<D> + MulOp<D> + NegOp<D> + PowOp<D> + SubOp<D> + 'static,
        D: Float + 'static,
    {
        let op = Box::new(autograd::PowBackward::new());
        self.record_binary_node(
            other,
            out,
            op,
            vec![self.clone(), other.clone(), out.clone()],
        );
        Ok(())
    }
}
