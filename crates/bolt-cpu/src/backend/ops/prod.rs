use std::ops::Mul;

use bolt_core::{
    StorageAllocator, TensorParts,
    dtype::{NativeType, OneValue},
    error::{Error, Result},
    layout::Layout,
    shape::{ConcreteShape, canonical_axes},
};

use super::super::allocator::CpuAllocator;
use super::super::storage::{CpuStorage, CpuTensorView};
use super::reduction_helpers::{
    compute_multi_index_from_linear, compute_output_linear_index, compute_reduction_shape,
};

pub trait ProdKernel: NativeType {
    fn prod_kernel(
        _view: CpuTensorView<'_, Self>,
        _axes: Option<&[isize]>,
        _keepdims: bool,
        _allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "prod not implemented for {}",
            Self::DTYPE
        )))
    }
}

fn reduce_prod<D>(
    view: CpuTensorView<'_, D>,
    axes: Option<&[isize]>,
    keepdims: bool,
    allocator: &CpuAllocator<D>,
) -> Result<TensorParts<CpuStorage<D>>>
where
    D: NativeType + Copy + Mul<Output = D> + OneValue,
{
    let view_shape = view.layout.shape();

    let canonical = axes
        .map(|ax| canonical_axes(ax, view_shape.len()))
        .transpose()?;
    let output_shape = compute_reduction_shape(view_shape, canonical.as_deref(), keepdims)?;

    let output_numel: usize = if output_shape.is_empty() {
        1
    } else {
        output_shape.iter().product()
    };

    let mut out_storage = allocator.allocate(output_numel)?;
    let out_data = out_storage.try_as_uninit_slice_mut()?;

    for slot in out_data.iter_mut().take(output_numel) {
        slot.write(D::one());
    }

    let view_data = view.storage.as_uninit_slice();

    if axes.is_none() {
        let mut prod = D::one();
        if view.layout.is_contiguous() && view.layout.offset_bytes() == 0 {
            let numel: usize = view_shape.iter().product();
            for slot in view_data.iter().take(numel) {
                prod = prod * unsafe { slot.assume_init() };
            }
        } else {
            for idx in view.layout.iter_element_indices(D::DTYPE)? {
                prod = prod * unsafe { view_data[idx].assume_init() };
            }
        }
        out_data[0].write(prod);
    } else {
        let canonical = canonical.unwrap();

        let mut logical_idx = 0;
        for idx in view.layout.iter_element_indices(D::DTYPE)? {
            let value = unsafe { view_data[idx].assume_init() };

            let input_indices = compute_multi_index_from_linear(logical_idx, view_shape);

            let output_linear_idx =
                compute_output_linear_index(&input_indices, &canonical, &output_shape, keepdims);

            let current = unsafe { out_data[output_linear_idx].assume_init_ref() };
            out_data[output_linear_idx].write(*current * value);

            logical_idx += 1;
        }
    }

    let out_layout = Layout::contiguous(ConcreteShape::from_slice(&output_shape)?);

    Ok(TensorParts {
        storage: out_storage,
        layout: out_layout,
    })
}

impl ProdKernel for f32 {
    fn prod_kernel(
        view: CpuTensorView<'_, Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        reduce_prod(view, axes, keepdims, allocator)
    }
}

impl ProdKernel for f64 {
    fn prod_kernel(
        view: CpuTensorView<'_, Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        reduce_prod(view, axes, keepdims, allocator)
    }
}

impl ProdKernel for i32 {
    fn prod_kernel(
        view: CpuTensorView<'_, Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        reduce_prod(view, axes, keepdims, allocator)
    }
}
