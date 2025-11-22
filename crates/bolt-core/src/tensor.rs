use std::{fmt, marker::PhantomData, sync::Arc};

use bytemuck::cast;

use crate::{
    allocator::StorageAllocator,
    backend::Backend,
    device::DeviceKind,
    dtype::{DType, NativeType, OneValue, ToF32},
    error::{Error, Result},
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

    pub fn zeros(backend: &Arc<B>, shape: &[usize]) -> Result<Self> {
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
    {
        Self::full(backend, shape, tensor_creation::one_value::<D>())
    }

    pub fn full(backend: &Arc<B>, shape: &[usize], value: D) -> Result<Self> {
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
}

const DISPLAY_EDGE_ITEMS: usize = 3;
const DISPLAY_TOTAL_ELEMENT_THRESHOLD: usize = 1000;
const DISPLAY_LINE_WIDTH: usize = 80;

enum DisplayIndex {
    Index(usize),
    Ellipsis,
}

struct TensorFormatter<'a, B, D>
where
    B: Backend<D>,
    D: NativeType,
{
    tensor: &'a Tensor<B, D>,
    scalar_shape: ConcreteShape,
}

impl<'a, B, D> TensorFormatter<'a, B, D>
where
    B: Backend<D>,
    D: NativeType,
{
    fn new(tensor: &'a Tensor<B, D>) -> Self {
        let scalar_shape =
            ConcreteShape::from_slice(&[1]).expect("scalar shape construction must succeed");
        Self {
            tensor,
            scalar_shape,
        }
    }

    fn format(&mut self) -> Result<String> {
        let truncated = self.tensor.numel() > DISPLAY_TOTAL_ELEMENT_THRESHOLD;
        let mut out = String::new();
        let mut indices = Vec::with_capacity(self.tensor.shape().len());
        self.format_dim(0, &mut indices, truncated, 0, &mut out)?;
        Ok(out)
    }

    fn format_dim(
        &mut self,
        dim: usize,
        indices: &mut Vec<usize>,
        truncated: bool,
        depth: usize,
        out: &mut String,
    ) -> Result<()> {
        out.push('[');
        let len = self.tensor.shape()[dim];
        let entries = self.display_indices(len, truncated);
        let rank = self.tensor.shape().len();
        for (i, entry) in entries.iter().enumerate() {
            if rank == dim + 1 {
                if i > 0 {
                    out.push_str(", ");
                }
                match entry {
                    DisplayIndex::Index(value_idx) => {
                        indices.push(*value_idx);
                        let value = self.read_value(indices)?;
                        out.push_str(&self.format_value(value));
                        indices.pop();
                    }
                    DisplayIndex::Ellipsis => out.push_str("..."),
                }
            } else {
                if i > 0 {
                    out.push(',');
                }
                if depth > 0 || i > 0 {
                    out.push('\n');
                    out.push_str(&indent(depth + 1));
                }
                match entry {
                    DisplayIndex::Index(value_idx) => {
                        indices.push(*value_idx);
                        self.format_dim(dim + 1, indices, truncated, depth + 1, out)?;
                        indices.pop();
                    }
                    DisplayIndex::Ellipsis => out.push_str("..."),
                }
            }
        }
        if rank != dim + 1 {
            out.push('\n');
            out.push_str(&indent(depth));
        }
        out.push(']');
        Ok(())
    }

    fn display_indices(&self, len: usize, truncated: bool) -> Vec<DisplayIndex> {
        if !truncated || len <= DISPLAY_EDGE_ITEMS * 2 {
            return (0..len).map(DisplayIndex::Index).collect();
        }
        let mut out = Vec::with_capacity(DISPLAY_EDGE_ITEMS * 2 + 1);
        for idx in 0..DISPLAY_EDGE_ITEMS {
            out.push(DisplayIndex::Index(idx));
        }
        out.push(DisplayIndex::Ellipsis);
        for idx in (len - DISPLAY_EDGE_ITEMS)..len {
            out.push(DisplayIndex::Index(idx));
        }
        out
    }

    fn read_value(&self, indices: &[usize]) -> Result<D> {
        let offset_bytes = self
            .tensor
            .layout
            .offset_bytes_for_indices(indices, D::DTYPE)?;
        let layout = Layout::contiguous_with_offset(self.scalar_shape.clone(), offset_bytes);
        let mut value = [D::default(); 1];
        self.tensor
            .backend
            .read(&self.tensor.storage, &layout, &mut value)?;
        Ok(value[0])
    }

    fn format_value(&self, value: D) -> String {
        match D::DTYPE {
            DType::I32 => {
                let v: i32 = cast(value);
                format!("{v}")
            }
            DType::F32 => {
                let v: f32 = cast(value);
                format_float(v as f64)
            }
            DType::F64 => {
                let v: f64 = cast(value);
                format_float(v)
            }
        }
    }
}

impl<B, D> fmt::Display for Tensor<B, D>
where
    B: Backend<D>,
    D: NativeType + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut formatter = TensorFormatter::new(self);
        let values = formatter.format().map_err(|_| fmt::Error);
        match values {
            Ok(rendered) => write!(
                f,
                "tensor({}, shape={}, dtype={}, device={})",
                rendered,
                format_shape(self.shape()),
                D::DTYPE,
                format_device(self.backend.device_kind())
            ),
            Err(_) => write!(
                f,
                "tensor(<unavailable: device read failed>, shape={}, dtype={}, device={})",
                format_shape(self.shape()),
                D::DTYPE,
                format_device(self.backend.device_kind())
            ),
        }
    }
}

impl<B, D> fmt::Debug for Tensor<B, D>
where
    B: Backend<D>,
    D: NativeType + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

fn format_shape(shape: &[usize]) -> String {
    let mut out = String::from("[");
    for (i, dim) in shape.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&dim.to_string());
    }
    out.push(']');
    out
}

fn format_device(kind: DeviceKind) -> &'static str {
    match kind {
        DeviceKind::Cpu => "cpu",
        DeviceKind::Cuda => "cuda",
    }
}

fn format_float(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-inf".into()
        } else {
            "inf".into()
        };
    }
    let abs = value.abs();
    let use_sci = abs != 0.0 && (abs < 1e-4 || abs >= 1e4);
    let mut repr = if use_sci {
        format!("{value:.6e}")
    } else {
        format!("{value:.6}")
    };
    if let Some(dot) = repr.find('.') {
        let mut end = repr.len();
        while end > dot + 1 && repr.as_bytes()[end - 1] == b'0' {
            end -= 1;
        }
        if end > dot && repr.as_bytes()[end - 1] == b'.' {
            end -= 1;
        }
        repr.truncate(end);
    }
    repr
}

fn indent(depth: usize) -> String {
    const SPACES_PER_LEVEL: usize = 2;
    let width = depth.saturating_mul(SPACES_PER_LEVEL);
    " ".repeat(width.min(DISPLAY_LINE_WIDTH))
}
