use bolt_core::{
    StorageAllocator, TensorParts,
    dtype::FloatType,
    error::{Error, Result},
    layout::Layout,
    shape::ConcreteShape,
};

use super::super::allocator::CpuAllocator;
use super::super::storage::{CpuStorage, CpuTensorView};

pub trait PowKernel: FloatType {
    fn pow_kernel(
        _lhs: CpuTensorView<'_, Self>,
        _rhs: CpuTensorView<'_, Self>,
        _alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "pow not implemented for {}",
            std::any::type_name::<Self>()
        )))
    }
}

pub fn pow<D>(
    lhs: CpuTensorView<'_, D>,
    rhs: CpuTensorView<'_, D>,
    allocator: &CpuAllocator<D>,
) -> Result<TensorParts<CpuStorage<D>>>
where
    D: FloatType + num_traits::Float,
{
    let (lhs_layout, rhs_layout) = Layout::broadcast_binary(lhs.layout, rhs.layout)?;
    let shape = lhs_layout.shape();
    let numel: usize = shape.iter().product();
    let mut out_storage: CpuStorage<D> = allocator.allocate(numel)?;
    let out_slice = out_storage.try_as_uninit_slice_mut()?;

    let lhs_data = lhs.storage.as_uninit_slice();
    let rhs_data = rhs.storage.as_uninit_slice();
    let elem_size = D::DTYPE.size_in_bytes();

    let no_broadcast = lhs.layout.shape() == rhs.layout.shape();
    if no_broadcast
        && lhs_layout.is_contiguous()
        && rhs_layout.is_contiguous()
        && lhs_layout.offset_bytes() == 0
        && rhs_layout.offset_bytes() == 0
    {
        for (dst, (&lhs_val, &rhs_val)) in out_slice
            .iter_mut()
            .zip(lhs_data.iter().zip(rhs_data.iter()))
        {
            let lhs_val = unsafe { lhs_val.assume_init() };
            let rhs_val = unsafe { rhs_val.assume_init() };
            dst.write(lhs_val.powf(rhs_val));
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
            dst.write(lhs_val.powf(rhs_val));
        }
    }

    let layout = Layout::contiguous(ConcreteShape::from_slice(shape)?);
    Ok(TensorParts {
        storage: out_storage,
        layout,
    })
}

impl PowKernel for f32 {
    fn pow_kernel(
        lhs: CpuTensorView<'_, Self>,
        rhs: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        pow(lhs, rhs, alloc)
    }
}

impl PowKernel for f64 {
    fn pow_kernel(
        lhs: CpuTensorView<'_, Self>,
        rhs: CpuTensorView<'_, Self>,
        alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        pow(lhs, rhs, alloc)
    }
}
