use std::{marker::PhantomData, sync::Arc};

use crate::{
    allocator::StorageAllocator,
    backend::Backend,
    dtype::NativeType,
    error::{Error, Result},
    layout::Layout,
    shape::ConcreteShape,
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
        Ok(Self::from_parts(backend.clone(), storage, layout))
    }

    pub fn zeros(backend: &Arc<B>, shape: &[usize]) -> Result<Self> {
        let shape = ConcreteShape::from_slice(shape)?;
        let numel = shape.num_elements();
        let layout = Layout::contiguous(shape);
        let storage = backend.allocator().allocate_zeroed(numel)?;
        Ok(Self::from_parts(backend.clone(), storage, layout))
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

    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        let shape = ConcreteShape::from_slice(shape)?;
        let layout = self.layout.reshape(shape)?;
        Ok(self.with_layout(layout))
    }

    pub fn slice(&self, axis: usize, start: usize, end: usize, step: usize) -> Result<Self> {
        let layout = self.layout.slice(axis, start, end, step, D::DTYPE)?;
        Ok(self.with_layout(layout))
    }

    pub fn permute(&self, axes: &[usize]) -> Result<Self> {
        let layout = self.layout.permute(axes)?;
        Ok(self.with_layout(layout))
    }

    pub fn transpose(&self, axis_a: usize, axis_b: usize) -> Result<Self> {
        let layout = self.layout.transpose(axis_a, axis_b)?;
        Ok(self.with_layout(layout))
    }

    pub fn contiguous(&self) -> Result<Self> {
        if self.layout.is_contiguous() && self.layout.offset_bytes() == 0 {
            return Ok(self.clone());
        }
        let (storage, layout) = self.backend.copy(&self.storage, &self.layout)?;
        Ok(Self::from_parts(self.backend.clone(), storage, layout))
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        self.ensure_same_backend(other)?;
        let (storage, layout) =
            self.backend
                .add(&self.storage, &other.storage, &self.layout, &other.layout)?;
        Ok(Self::from_parts(self.backend.clone(), storage, layout))
    }

    pub fn sub(&self, other: &Self) -> Result<Self> {
        self.ensure_same_backend(other)?;
        let (storage, layout) =
            self.backend
                .sub(&self.storage, &other.storage, &self.layout, &other.layout)?;
        Ok(Self::from_parts(self.backend.clone(), storage, layout))
    }

    pub fn matmul(&self, other: &Self) -> Result<Self> {
        self.ensure_same_backend(other)?;
        let (storage, layout) =
            self.backend
                .matmul(&self.storage, &other.storage, &self.layout, &other.layout)?;
        Ok(Self::from_parts(self.backend.clone(), storage, layout))
    }

    fn ensure_same_backend(&self, other: &Self) -> Result<()> {
        if !Arc::ptr_eq(&self.backend, &other.backend) {
            return Err(Error::Device("tensors belong to different backends".into()));
        }
        Ok(())
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
}
