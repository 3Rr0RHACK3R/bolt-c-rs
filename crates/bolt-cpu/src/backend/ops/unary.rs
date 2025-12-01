use std::ops::Neg;

use bolt_core::{
    StorageAllocator, TensorParts,
    dtype::NativeType,
    error::{Error, Result},
    layout::Layout,
    shape::ConcreteShape,
};

use super::super::allocator::CpuAllocator;
use super::super::storage::{CpuStorage, CpuTensorView};

pub trait NegKernel: NativeType {
    fn neg_kernel(
        _input: CpuTensorView<'_, Self>,
        _alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "neg not implemented for {}",
            std::any::type_name::<Self>()
        )))
    }
}

pub trait AbsKernel: NativeType {
    fn abs_kernel(
        _input: CpuTensorView<'_, Self>,
        _alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "abs not implemented for {}",
            std::any::type_name::<Self>()
        )))
    }
}

pub trait ExpKernel: NativeType {
    fn exp_kernel(
        _input: CpuTensorView<'_, Self>,
        _alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "exp not implemented for {}",
            std::any::type_name::<Self>()
        )))
    }
}

pub trait LogKernel: NativeType {
    fn log_kernel(
        _input: CpuTensorView<'_, Self>,
        _alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "log not implemented for {}",
            std::any::type_name::<Self>()
        )))
    }
}

pub trait SqrtKernel: NativeType {
    fn sqrt_kernel(
        _input: CpuTensorView<'_, Self>,
        _alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "sqrt not implemented for {}",
            std::any::type_name::<Self>()
        )))
    }
}

pub trait SinKernel: NativeType {
    fn sin_kernel(
        _input: CpuTensorView<'_, Self>,
        _alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "sin not implemented for {}",
            std::any::type_name::<Self>()
        )))
    }
}

pub trait CosKernel: NativeType {
    fn cos_kernel(
        _input: CpuTensorView<'_, Self>,
        _alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "cos not implemented for {}",
            std::any::type_name::<Self>()
        )))
    }
}

pub trait TanhKernel: NativeType {
    fn tanh_kernel(
        _input: CpuTensorView<'_, Self>,
        _alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "tanh not implemented for {}",
            std::any::type_name::<Self>()
        )))
    }
}

pub trait ReluKernel: NativeType {
    fn relu_kernel(
        _input: CpuTensorView<'_, Self>,
        _alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "relu not implemented for {}",
            std::any::type_name::<Self>()
        )))
    }
}

fn neg<D>(input: CpuTensorView<'_, D>, allocator: &CpuAllocator<D>) -> Result<TensorParts<CpuStorage<D>>>
where
    D: NativeType + Neg<Output = D>,
{
    let shape = input.layout.shape();
    let numel: usize = shape.iter().product();
    let mut out_storage: CpuStorage<D> = allocator.allocate(numel)?;
    let out_slice = out_storage.try_as_uninit_slice_mut()?;
    let input_data = input.storage.as_uninit_slice();
    let elem_size = D::DTYPE.size_in_bytes();

    if input.layout.is_contiguous() && input.layout.offset_bytes() == 0 {
        for (dst, &val) in out_slice.iter_mut().zip(input_data.iter()) {
            let val = unsafe { val.assume_init() };
            dst.write(-val);
        }
    } else {
        let iter = input.layout.iter_offsets(D::DTYPE)?;
        for (dst, idx_bytes) in out_slice.iter_mut().zip(iter) {
            debug_assert_eq!(idx_bytes % elem_size, 0);
            let idx = idx_bytes / elem_size;
            let val = unsafe { input_data[idx].assume_init() };
            dst.write(-val);
        }
    }

    let layout = Layout::contiguous(ConcreteShape::from_slice(shape)?);
    Ok(TensorParts {
        storage: out_storage,
        layout,
    })
}

fn abs<D>(input: CpuTensorView<'_, D>, allocator: &CpuAllocator<D>) -> Result<TensorParts<CpuStorage<D>>>
where
    D: NativeType + PartialOrd + Neg<Output = D>,
{
    let shape = input.layout.shape();
    let numel: usize = shape.iter().product();
    let mut out_storage: CpuStorage<D> = allocator.allocate(numel)?;
    let out_slice = out_storage.try_as_uninit_slice_mut()?;
    let input_data = input.storage.as_uninit_slice();
    let elem_size = D::DTYPE.size_in_bytes();
    let zero = D::default();

    if input.layout.is_contiguous() && input.layout.offset_bytes() == 0 {
        for (dst, &val) in out_slice.iter_mut().zip(input_data.iter()) {
            let val = unsafe { val.assume_init() };
            dst.write(if val < zero { -val } else { val });
        }
    } else {
        let iter = input.layout.iter_offsets(D::DTYPE)?;
        for (dst, idx_bytes) in out_slice.iter_mut().zip(iter) {
            debug_assert_eq!(idx_bytes % elem_size, 0);
            let idx = idx_bytes / elem_size;
            let val = unsafe { input_data[idx].assume_init() };
            dst.write(if val < zero { -val } else { val });
        }
    }

    let layout = Layout::contiguous(ConcreteShape::from_slice(shape)?);
    Ok(TensorParts {
        storage: out_storage,
        layout,
    })
}

macro_rules! impl_float_unary {
    ($fn_name:ident, $method:ident) => {
        fn $fn_name<D>(
            input: CpuTensorView<'_, D>,
            allocator: &CpuAllocator<D>,
        ) -> Result<TensorParts<CpuStorage<D>>>
        where
            D: NativeType,
            D: num_traits::Float,
        {
            let shape = input.layout.shape();
            let numel: usize = shape.iter().product();
            let mut out_storage: CpuStorage<D> = allocator.allocate(numel)?;
            let out_slice = out_storage.try_as_uninit_slice_mut()?;
            let input_data = input.storage.as_uninit_slice();
            let elem_size = D::DTYPE.size_in_bytes();

            if input.layout.is_contiguous() && input.layout.offset_bytes() == 0 {
                for (dst, &val) in out_slice.iter_mut().zip(input_data.iter()) {
                    let val = unsafe { val.assume_init() };
                    dst.write(val.$method());
                }
            } else {
                let iter = input.layout.iter_offsets(D::DTYPE)?;
                for (dst, idx_bytes) in out_slice.iter_mut().zip(iter) {
                    debug_assert_eq!(idx_bytes % elem_size, 0);
                    let idx = idx_bytes / elem_size;
                    let val = unsafe { input_data[idx].assume_init() };
                    dst.write(val.$method());
                }
            }

            let layout = Layout::contiguous(ConcreteShape::from_slice(shape)?);
            Ok(TensorParts {
                storage: out_storage,
                layout,
            })
        }
    };
}

impl_float_unary!(exp, exp);
impl_float_unary!(log, ln);
impl_float_unary!(sqrt, sqrt);
impl_float_unary!(sin, sin);
impl_float_unary!(cos, cos);
impl_float_unary!(tanh, tanh);

fn relu<D>(input: CpuTensorView<'_, D>, allocator: &CpuAllocator<D>) -> Result<TensorParts<CpuStorage<D>>>
where
    D: NativeType + PartialOrd,
{
    let shape = input.layout.shape();
    let numel: usize = shape.iter().product();
    let mut out_storage: CpuStorage<D> = allocator.allocate(numel)?;
    let out_slice = out_storage.try_as_uninit_slice_mut()?;
    let input_data = input.storage.as_uninit_slice();
    let elem_size = D::DTYPE.size_in_bytes();
    let zero = D::default();

    if input.layout.is_contiguous() && input.layout.offset_bytes() == 0 {
        for (dst, &val) in out_slice.iter_mut().zip(input_data.iter()) {
            let val = unsafe { val.assume_init() };
            dst.write(if val > zero { val } else { zero });
        }
    } else {
        let iter = input.layout.iter_offsets(D::DTYPE)?;
        for (dst, idx_bytes) in out_slice.iter_mut().zip(iter) {
            debug_assert_eq!(idx_bytes % elem_size, 0);
            let idx = idx_bytes / elem_size;
            let val = unsafe { input_data[idx].assume_init() };
            dst.write(if val > zero { val } else { zero });
        }
    }

    let layout = Layout::contiguous(ConcreteShape::from_slice(shape)?);
    Ok(TensorParts {
        storage: out_storage,
        layout,
    })
}

impl NegKernel for f32 {
    fn neg_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        neg(input, alloc)
    }
}

impl NegKernel for f64 {
    fn neg_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        neg(input, alloc)
    }
}

impl NegKernel for i32 {
    fn neg_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        neg(input, alloc)
    }
}

impl AbsKernel for f32 {
    fn abs_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        abs(input, alloc)
    }
}

impl AbsKernel for f64 {
    fn abs_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        abs(input, alloc)
    }
}

impl AbsKernel for i32 {
    fn abs_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        abs(input, alloc)
    }
}

impl ExpKernel for f32 {
    fn exp_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        exp(input, alloc)
    }
}

impl ExpKernel for f64 {
    fn exp_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        exp(input, alloc)
    }
}

impl LogKernel for f32 {
    fn log_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        log(input, alloc)
    }
}

impl LogKernel for f64 {
    fn log_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        log(input, alloc)
    }
}

impl SqrtKernel for f32 {
    fn sqrt_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        sqrt(input, alloc)
    }
}

impl SqrtKernel for f64 {
    fn sqrt_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        sqrt(input, alloc)
    }
}

impl SinKernel for f32 {
    fn sin_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        sin(input, alloc)
    }
}

impl SinKernel for f64 {
    fn sin_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        sin(input, alloc)
    }
}

impl CosKernel for f32 {
    fn cos_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        cos(input, alloc)
    }
}

impl CosKernel for f64 {
    fn cos_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        cos(input, alloc)
    }
}

impl TanhKernel for f32 {
    fn tanh_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        tanh(input, alloc)
    }
}

impl TanhKernel for f64 {
    fn tanh_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        tanh(input, alloc)
    }
}

impl ReluKernel for f32 {
    fn relu_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        relu(input, alloc)
    }
}

impl ReluKernel for f64 {
    fn relu_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        relu(input, alloc)
    }
}

impl ReluKernel for i32 {
    fn relu_kernel(
        input: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        relu(input, alloc)
    }
}

impl ExpKernel for i32 {}
impl LogKernel for i32 {}
impl SqrtKernel for i32 {}
impl SinKernel for i32 {}
impl CosKernel for i32 {}
impl TanhKernel for i32 {}
