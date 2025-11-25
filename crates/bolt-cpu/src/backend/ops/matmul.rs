use std::ops::{Add, Mul};

use bolt_core::{
    StorageAllocator, TensorParts,
    dtype::NativeType,
    error::{Error, Result},
    layout::Layout,
    shape::ConcreteShape,
};

use super::super::allocator::CpuAllocator;
use super::super::storage::{CpuStorage, CpuTensorView};

pub trait MatmulKernel: NativeType {
    fn matmul_kernel(
        _lhs: CpuTensorView<'_, Self>,
        _rhs: CpuTensorView<'_, Self>,
        _alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "matmul not implemented for {}",
            std::any::type_name::<Self>()
        )))
    }
}

pub fn matmul<D>(
    lhs: CpuTensorView<'_, D>,
    rhs: CpuTensorView<'_, D>,
    allocator: &CpuAllocator<D>,
) -> Result<TensorParts<CpuStorage<D>>>
where
    D: NativeType + Copy + Default + Add<Output = D> + Mul<Output = D>,
{
    let lhs_shape = lhs.layout.shape();
    let rhs_shape = rhs.layout.shape();
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
    let lhs_data = lhs.storage.as_uninit_slice();
    let rhs_data = rhs.storage.as_uninit_slice();
    let elem_size = D::DTYPE.size_in_bytes();

    let mut out_storage: CpuStorage<D> = allocator.allocate(m * n)?;
    let out_slice = out_storage.try_as_uninit_slice_mut()?;

    if lhs.layout.is_contiguous()
        && rhs.layout.is_contiguous()
        && lhs.layout.offset_bytes() == 0
        && rhs.layout.offset_bytes() == 0
        && lhs.layout.strides()[1] == 1
        && rhs.layout.strides()[1] == 1
    {
        let lhs_s0 = lhs.layout.strides()[0] as usize;
        let rhs_s0 = rhs.layout.strides()[0] as usize;

        for i in 0..m {
            for j in 0..n {
                let mut sum = D::default();
                for p in 0..k {
                    let lhs_idx = i * lhs_s0 + p;
                    let rhs_idx = p * rhs_s0 + j;
                    let lhs_val = unsafe { lhs_data[lhs_idx].assume_init() };
                    let rhs_val = unsafe { rhs_data[rhs_idx].assume_init() };
                    sum = sum + lhs_val * rhs_val;
                }
                out_slice[i * n + j].write(sum);
            }
        }
    } else {
        let rhs_col_offsets: Vec<Vec<usize>> = (0..n)
            .map(|j| {
                let layout = rhs.layout.slice(1, j, j + 1, 1, D::DTYPE)?;
                let mut offsets = Vec::with_capacity(k);
                for idx_bytes in layout.iter_offsets(D::DTYPE)? {
                    debug_assert_eq!(idx_bytes % elem_size, 0);
                    offsets.push(idx_bytes / elem_size);
                }
                Ok(offsets)
            })
            .collect::<Result<_>>()?;

        for i in 0..m {
            let lhs_row_layout = lhs.layout.slice(0, i, i + 1, 1, D::DTYPE)?;
            let mut lhs_offsets = Vec::with_capacity(k);
            for idx_bytes in lhs_row_layout.iter_offsets(D::DTYPE)? {
                debug_assert_eq!(idx_bytes % elem_size, 0);
                lhs_offsets.push(idx_bytes / elem_size);
            }

            for (j, rhs_offsets) in rhs_col_offsets.iter().enumerate() {
                let mut sum = D::default();
                debug_assert_eq!(lhs_offsets.len(), rhs_offsets.len());

                for (lhs_idx, rhs_idx) in lhs_offsets.iter().zip(rhs_offsets.iter()) {
                    let lhs_val = unsafe { lhs_data[*lhs_idx].assume_init() };
                    let rhs_val = unsafe { rhs_data[*rhs_idx].assume_init() };
                    sum = sum + lhs_val * rhs_val;
                }
                out_slice[i * n + j].write(sum);
            }
        }
    }

    let layout = Layout::contiguous(ConcreteShape::from_slice(&[m, n])?);
    Ok(TensorParts {
        storage: out_storage,
        layout,
    })
}

impl MatmulKernel for f32 {
    fn matmul_kernel(
        _lhs: CpuTensorView<'_, Self>,
        _rhs: CpuTensorView<'_, Self>,
        _alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        matmul(_lhs, _rhs, _alloc)
    }
}

impl MatmulKernel for f64 {
    fn matmul_kernel(
        _lhs: CpuTensorView<'_, Self>,
        _rhs: CpuTensorView<'_, Self>,
        _alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        matmul(_lhs, _rhs, _alloc)
    }
}

impl MatmulKernel for i32 {
    fn matmul_kernel(
        _lhs: CpuTensorView<'_, Self>,
        _rhs: CpuTensorView<'_, Self>,
        _alloc: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        matmul(_lhs, _rhs, _alloc)
    }
}
