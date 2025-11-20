use std::{marker::PhantomData, ops::Deref, sync::Arc};

use bolt_core::{
    allocator::{AllocatorHandle, AllocatorMetrics, StorageAllocator, StorageBlock},
    backend::{Backend, MeanOp},
    device::{BackendDevice, DeviceKind},
    dtype::{NativeType, TensorNum},
    error::{Error, Result},
    layout::Layout,
    shape::{ConcreteShape, broadcast_shapes},
};

type CpuStorage<D> = Arc<StorageBlock<D>>;

pub trait ToF32 {
    fn to_f32(self) -> f32;
}

impl ToF32 for f32 {
    fn to_f32(self) -> f32 {
        self
    }
}

impl ToF32 for f64 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl ToF32 for i32 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

#[derive(Clone)]
pub struct CpuBackend {
    device: Arc<CpuDevice>,
    allocators: CpuAllocators,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            device: Arc::new(CpuDevice),
            allocators: CpuAllocators::new(),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
struct CpuAllocators {
    f32: CpuAllocator<f32>,
    f64: CpuAllocator<f64>,
    i32: CpuAllocator<i32>,
}

impl CpuAllocators {
    fn new() -> Self {
        Self {
            f32: CpuAllocator::new(),
            f64: CpuAllocator::new(),
            i32: CpuAllocator::new(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CpuDevice;

impl BackendDevice for CpuDevice {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }
}

#[derive(Clone, Debug)]
pub struct CpuAllocator<D: NativeType> {
    handle: AllocatorHandle,
    _marker: PhantomData<D>,
}

impl<D: NativeType> CpuAllocator<D> {
    fn new() -> Self {
        Self {
            handle: AllocatorHandle::new(),
            _marker: PhantomData,
        }
    }
}

impl<D: NativeType> StorageAllocator<D> for CpuAllocator<D> {
    type Storage = CpuStorage<D>;

    fn allocate(&self, len: usize) -> Result<Self::Storage> {
        Ok(Arc::new(StorageBlock::new(len, &self.handle, false)))
    }

    fn allocate_zeroed(&self, len: usize) -> Result<Self::Storage> {
        Ok(Arc::new(StorageBlock::new(len, &self.handle, true)))
    }

    fn metrics(&self) -> AllocatorMetrics {
        self.handle.snapshot()
    }
}

macro_rules! impl_backend_for {
    ($dtype:ty, $field:ident) => {
        impl Backend<$dtype> for CpuBackend {
            type Device = CpuDevice;
            type Storage = CpuStorage<$dtype>;
            type Allocator = CpuAllocator<$dtype>;

            fn device(&self) -> &Self::Device {
                &self.device
            }

            fn allocator(&self) -> &Self::Allocator {
                &self.allocators.$field
            }

            fn read(
                &self,
                storage: &Self::Storage,
                layout: &Layout,
                dst: &mut [$dtype],
            ) -> Result<()> {
                read_into_slice(storage, layout, dst)
            }

            fn write(
                &self,
                storage: &mut Self::Storage,
                layout: &Layout,
                src: &[$dtype],
            ) -> Result<()> {
                write_from_slice(storage, layout, src)
            }

            fn copy(
                &self,
                storage: &Self::Storage,
                layout: &Layout,
            ) -> Result<(Self::Storage, Layout)> {
                copy_to_contiguous(storage, layout, &self.allocators.$field)
            }

            fn add(
                &self,
                lhs: &Self::Storage,
                rhs: &Self::Storage,
                lhs_layout: &Layout,
                rhs_layout: &Layout,
            ) -> Result<(Self::Storage, Layout)> {
                binary_op(
                    lhs,
                    rhs,
                    lhs_layout,
                    rhs_layout,
                    &self.allocators.$field,
                    |a, b| a + b,
                )
            }

            fn sub(
                &self,
                lhs: &Self::Storage,
                rhs: &Self::Storage,
                lhs_layout: &Layout,
                rhs_layout: &Layout,
            ) -> Result<(Self::Storage, Layout)> {
                binary_op(
                    lhs,
                    rhs,
                    lhs_layout,
                    rhs_layout,
                    &self.allocators.$field,
                    |a, b| a - b,
                )
            }

            fn matmul(
                &self,
                lhs: &Self::Storage,
                rhs: &Self::Storage,
                lhs_layout: &Layout,
                rhs_layout: &Layout,
            ) -> Result<(Self::Storage, Layout)> {
                matmul(lhs, rhs, lhs_layout, rhs_layout, &self.allocators.$field)
            }
        }
    };
}

impl_backend_for!(f32, f32);
impl_backend_for!(f64, f64);
impl_backend_for!(i32, i32);

impl<D> MeanOp<D, f32> for CpuBackend
where
    D: ToF32 + NativeType,
    CpuBackend: Backend<D>,
    <CpuBackend as Backend<D>>::Storage: Deref<Target = StorageBlock<D>>,
{
    fn mean(
        &self,
        storage: &<Self as Backend<D>>::Storage,
        layout: &Layout,
    ) -> Result<(<Self as Backend<f32>>::Storage, Layout)> {
        let numel = layout.num_elements();
        if numel == 0 {
            return Err(Error::InvalidShape {
                message: "mean requires at least one element".into(),
            });
        }

        let shape = layout.shape();
        let strides = expand_strides(layout, shape)?;
        let data = storage.deref().as_slice();
        let offset = layout.offset_elements(D::DTYPE);
        let mut coords = vec![0usize; shape.len()];
        let mut acc = 0f32;
        for idx in 0..numel {
            linear_to_indices(idx, shape, &mut coords);
            let src_idx = offset + offset_from_strides(&coords, &strides);
            acc += data[src_idx as usize].to_f32();
        }
        let mean = acc / numel as f32;

        let mut output = self.allocators.f32.allocate(1)?;
        {
            let slice = Arc::get_mut(&mut output)
                .expect("unique storage")
                .as_mut_slice();
            slice[0] = mean;
        }
        let layout = Layout::contiguous(ConcreteShape::from_slice(&[1])?);
        Ok((output, layout))
    }
}

fn read_into_slice<D: NativeType>(
    storage: &CpuStorage<D>,
    layout: &Layout,
    dst: &mut [D],
) -> Result<()> {
    if layout.num_elements() != dst.len() {
        return Err(Error::SizeMismatch {
            expected: layout.num_elements(),
            actual: dst.len(),
        });
    }
    if layout.is_contiguous() && layout.offset_bytes() == 0 {
        let slice = storage.as_slice();
        dst.copy_from_slice(&slice[..dst.len()]);
        return Ok(());
    }
    let shape = layout.shape();
    let strides = expand_strides(layout, shape)?;
    let offset = layout.offset_elements(D::DTYPE);
    let data = storage.as_slice();
    let mut coords = vec![0usize; shape.len()];
    for idx in 0..layout.num_elements() {
        linear_to_indices(idx, shape, &mut coords);
        let src = offset + offset_from_strides(&coords, &strides);
        dst[idx] = data[src as usize];
    }
    Ok(())
}

fn write_from_slice<D: NativeType>(
    storage: &mut CpuStorage<D>,
    layout: &Layout,
    src: &[D],
) -> Result<()> {
    if layout.num_elements() != src.len() {
        return Err(Error::SizeMismatch {
            expected: layout.num_elements(),
            actual: src.len(),
        });
    }
    let block = Arc::get_mut(storage).ok_or_else(|| {
        Error::OpError("cannot write to shared tensor storage; clone before mutating".into())
    })?;
    if layout.is_contiguous() && layout.offset_bytes() == 0 {
        let dst = block.as_mut_slice();
        dst.copy_from_slice(src);
        return Ok(());
    }
    let shape = layout.shape();
    let strides = expand_strides(layout, shape)?;
    let offset = layout.offset_elements(D::DTYPE);
    let dst = block.as_mut_slice();
    let mut coords = vec![0usize; shape.len()];
    for idx in 0..layout.num_elements() {
        linear_to_indices(idx, shape, &mut coords);
        let dst_idx = offset + offset_from_strides(&coords, &strides);
        dst[dst_idx as usize] = src[idx];
    }
    Ok(())
}

fn copy_to_contiguous<D: NativeType>(
    storage: &CpuStorage<D>,
    layout: &Layout,
    allocator: &CpuAllocator<D>,
) -> Result<(CpuStorage<D>, Layout)> {
    let shape = ConcreteShape::from_slice(layout.shape())?;
    let mut dst = allocator.allocate(layout.num_elements())?;
    {
        let block = Arc::get_mut(&mut dst).expect("unique storage");
        read_into_slice(storage, layout, block.as_mut_slice())?;
    }
    let layout = Layout::contiguous(shape);
    Ok((dst, layout))
}

fn binary_op<D: TensorNum>(
    lhs_storage: &CpuStorage<D>,
    rhs_storage: &CpuStorage<D>,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    allocator: &CpuAllocator<D>,
    func: impl Fn(D, D) -> D,
) -> Result<(CpuStorage<D>, Layout)> {
    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    let shape = broadcast_shapes(lhs_shape, rhs_shape)?;
    let numel: usize = shape.iter().product();
    let mut out_storage = allocator.allocate(numel)?;
    let out_data = Arc::get_mut(&mut out_storage).expect("unique storage");
    let out_slice = out_data.as_mut_slice();

    let lhs_strides = expand_strides(lhs_layout, &shape)?;
    let rhs_strides = expand_strides(rhs_layout, &shape)?;
    let lhs_data = lhs_storage.as_slice();
    let rhs_data = rhs_storage.as_slice();
    let lhs_offset = lhs_layout.offset_elements(D::DTYPE);
    let rhs_offset = rhs_layout.offset_elements(D::DTYPE);

    let mut coords = vec![0usize; shape.len()];
    for i in 0..numel {
        linear_to_indices(i, &shape, &mut coords);
        let lhs_idx = lhs_offset + offset_from_strides(&coords, &lhs_strides);
        let rhs_idx = rhs_offset + offset_from_strides(&coords, &rhs_strides);
        out_slice[i] = func(lhs_data[lhs_idx as usize], rhs_data[rhs_idx as usize]);
    }

    let layout = Layout::contiguous(ConcreteShape::from_slice(&shape)?);
    Ok((out_storage, layout))
}

fn matmul<D: TensorNum>(
    lhs_storage: &CpuStorage<D>,
    rhs_storage: &CpuStorage<D>,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    allocator: &CpuAllocator<D>,
) -> Result<(CpuStorage<D>, Layout)> {
    let lhs_shape = lhs_layout.shape();
    let rhs_shape = rhs_layout.shape();
    if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
        return Err(Error::invalid_shape("matmul requires rank-2 inputs"));
    }
    let m = lhs_shape[0];
    let k = lhs_shape[1];
    if rhs_shape[0] != k {
        return Err(Error::ShapeMismatch {
            lhs: lhs_shape.to_vec(),
            rhs: rhs_shape.to_vec(),
        });
    }
    let n = rhs_shape[1];
    let lhs_data = lhs_storage.as_slice();
    let rhs_data = rhs_storage.as_slice();
    let lhs_s0 = lhs_layout.strides()[0];
    let lhs_s1 = lhs_layout.strides()[1];
    let rhs_s0 = rhs_layout.strides()[0];
    let rhs_s1 = rhs_layout.strides()[1];
    let lhs_off = lhs_layout.offset_elements(D::DTYPE);
    let rhs_off = rhs_layout.offset_elements(D::DTYPE);

    let mut out_storage = allocator.allocate(m * n)?;
    let out_slice = Arc::get_mut(&mut out_storage)
        .expect("unique storage")
        .as_mut_slice();

    for i in 0..m {
        for j in 0..n {
            let mut sum = D::default();
            for p in 0..k {
                let lhs_idx = lhs_off + (i as isize * lhs_s0) + (p as isize * lhs_s1);
                let rhs_idx = rhs_off + (p as isize * rhs_s0) + (j as isize * rhs_s1);
                sum = sum + lhs_data[lhs_idx as usize] * rhs_data[rhs_idx as usize];
            }
            out_slice[i * n + j] = sum;
        }
    }

    let layout = Layout::contiguous(ConcreteShape::from_slice(&[m, n])?);
    Ok((out_storage, layout))
}

fn linear_to_indices(index: usize, shape: &[usize], coords: &mut [usize]) {
    let mut value = index;
    for (axis, &dim) in shape.iter().enumerate().rev() {
        coords[axis] = value % dim;
        value /= dim;
    }
}

fn offset_from_strides(indices: &[usize], strides: &[isize]) -> isize {
    indices
        .iter()
        .zip(strides.iter())
        .map(|(&idx, &stride)| idx as isize * stride)
        .sum()
}

fn expand_strides(layout: &Layout, target_shape: &[usize]) -> Result<Vec<isize>> {
    let in_shape = layout.shape();
    let in_strides = layout.strides();
    let mut result = vec![0isize; target_shape.len()];
    let mut in_idx = in_shape.len();
    for out_idx in (0..target_shape.len()).rev() {
        let out_dim = target_shape[out_idx];
        if in_idx > 0 {
            let current_in = in_idx - 1;
            let in_dim = in_shape[current_in];
            if in_dim == out_dim {
                result[out_idx] = in_strides[current_in];
                in_idx -= 1;
            } else if in_dim == 1 {
                result[out_idx] = 0;
                in_idx -= 1;
            } else {
                return Err(Error::ShapeMismatch {
                    lhs: in_shape.to_vec(),
                    rhs: target_shape.to_vec(),
                });
            }
        } else {
            result[out_idx] = 0;
        }
    }
    Ok(result)
}
