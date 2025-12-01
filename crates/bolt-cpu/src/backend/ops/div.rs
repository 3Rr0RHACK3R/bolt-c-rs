use std::ops::Div;

use bolt_core::{
    StorageAllocator, TensorParts,
    dtype::NativeType,
    error::{Error, Result},
    layout::Layout,
    shape::ConcreteShape,
};

use super::super::allocator::CpuAllocator;
use super::super::storage::{CpuStorage, CpuTensorView};

pub trait DivKernel: NativeType {
    fn div_kernel(
        _lhs: CpuTensorView<'_, Self>,
        _rhs: CpuTensorView<'_, Self>,
        _alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "div not implemented for {}",
            std::any::type_name::<Self>()
        )))
    }
}

pub(crate) trait DivValidate: NativeType {
    fn check_divisor(val: Self) -> Result<()>;
}

impl DivValidate for f32 {
    #[inline(always)]
    fn check_divisor(_val: Self) -> Result<()> {
        Ok(())
    }
}

impl DivValidate for f64 {
    #[inline(always)]
    fn check_divisor(_val: Self) -> Result<()> {
        Ok(())
    }
}

impl DivValidate for i32 {
    #[inline(always)]
    fn check_divisor(val: Self) -> Result<()> {
        if val == 0 {
            Err(Error::OpError("division by zero".to_string()))
        } else {
            Ok(())
        }
    }
}

pub(crate) trait DivOverflowCheck: DivValidate {
    fn check_division(lhs: Self, rhs: Self) -> Result<()>;
}

impl DivOverflowCheck for f32 {
    #[inline(always)]
    fn check_division(_lhs: Self, rhs: Self) -> Result<()> {
        Self::check_divisor(rhs)
    }
}

impl DivOverflowCheck for f64 {
    #[inline(always)]
    fn check_division(_lhs: Self, rhs: Self) -> Result<()> {
        Self::check_divisor(rhs)
    }
}

impl DivOverflowCheck for i32 {
    #[inline(always)]
    fn check_division(lhs: Self, rhs: Self) -> Result<()> {
        Self::check_divisor(rhs)?;
        if lhs == i32::MIN && rhs == -1 {
            return Err(Error::OpError("integer overflow: i32::MIN / -1".to_string()));
        }
        Ok(())
    }
}

pub fn div<D>(
    lhs: CpuTensorView<'_, D>,
    rhs: CpuTensorView<'_, D>,
    allocator: &CpuAllocator<D>,
) -> Result<TensorParts<CpuStorage<D>>>
where
    D: NativeType + Copy + Div<Output = D> + DivOverflowCheck,
{
    let (lhs_layout, rhs_layout) = Layout::broadcast_binary(lhs.layout, rhs.layout)?;
    let shape = lhs_layout.shape();
    let numel: usize = shape.iter().product();
    let mut out_storage: CpuStorage<D> = allocator.allocate(numel)?;
    let out_slice = out_storage.try_as_uninit_slice_mut()?;

    let lhs_data = lhs.storage.as_uninit_slice();
    let rhs_data = rhs.storage.as_uninit_slice();
    let elem_size = D::DTYPE.size_in_bytes();

    let no_broadcast = lhs_layout.shape() == rhs_layout.shape()
        && lhs_layout.shape() == lhs.layout.shape()
        && rhs_layout.shape() == rhs.layout.shape();
    if no_broadcast
        && lhs.layout.is_contiguous()
        && rhs.layout.is_contiguous()
        && lhs.layout.offset_bytes() == 0
        && rhs.layout.offset_bytes() == 0
    {
        for (dst, (&lhs_val, &rhs_val)) in out_slice
            .iter_mut()
            .zip(lhs_data.iter().zip(rhs_data.iter()))
        {
            let lhs_val = unsafe { lhs_val.assume_init() };
            let rhs_val = unsafe { rhs_val.assume_init() };
            D::check_division(lhs_val, rhs_val)?;
            dst.write(lhs_val / rhs_val);
        }
    } else {
        let lhs_iter = lhs_layout.iter_offsets(D::DTYPE)?;
        let rhs_iter = rhs_layout.iter_offsets(D::DTYPE)?;
        for (dst, (lhs_idx_bytes, rhs_idx_bytes)) in
            out_slice.iter_mut().zip(lhs_iter.zip(rhs_iter))
        {
            debug_assert_eq!(lhs_idx_bytes % elem_size, 0);
            debug_assert_eq!(rhs_idx_bytes % elem_size, 0);
            let lhs_idx = lhs_idx_bytes / elem_size;
            let rhs_idx = rhs_idx_bytes / elem_size;
            let lhs_val = unsafe { lhs_data[lhs_idx].assume_init() };
            let rhs_val = unsafe { rhs_data[rhs_idx].assume_init() };
            D::check_division(lhs_val, rhs_val)?;
            dst.write(lhs_val / rhs_val);
        }
    }

    let layout = Layout::contiguous(ConcreteShape::from_slice(shape)?);
    Ok(TensorParts {
        storage: out_storage,
        layout,
    })
}

impl DivKernel for f32 {
    fn div_kernel(
        lhs: CpuTensorView<'_, Self>,
        rhs: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        div(lhs, rhs, alloc)
    }
}

impl DivKernel for f64 {
    fn div_kernel(
        lhs: CpuTensorView<'_, Self>,
        rhs: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        div(lhs, rhs, alloc)
    }
}

impl DivKernel for i32 {
    fn div_kernel(
        lhs: CpuTensorView<'_, Self>,
        rhs: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        div(lhs, rhs, alloc)
    }
}
