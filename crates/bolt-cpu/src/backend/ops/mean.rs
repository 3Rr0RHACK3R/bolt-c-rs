use bolt_core::{
    Float, StorageAllocator, TensorParts,
    error::{Error, Result},
    layout::Layout,
    shape::{Shape, canonical_axes},
};

use super::super::allocator::CpuAllocator;
use super::super::storage::{CpuStorage, CpuTensorView};
use super::reduction_helpers::{
    compute_multi_index_from_linear, compute_output_linear_index, compute_reduction_shape,
};

pub trait MeanKernel: Float {
    fn mean_kernel(
        _view: CpuTensorView<'_, Self>,
        _axes: Option<&[isize]>,
        _keepdims: bool,
        _allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "mean not implemented for {}",
            Self::DTYPE
        )))
    }
}

fn reduce_mean<D>(
    view: CpuTensorView<'_, D>,
    axes: Option<&[isize]>,
    keepdims: bool,
    allocator: &CpuAllocator<D>,
) -> Result<TensorParts<CpuStorage<D>>>
where
    D: Float,
{
    let view_shape = view.layout.shape();

    let canonical = axes
        .map(|ax| canonical_axes(ax, view_shape.len()))
        .transpose()?;
    let output_shape = compute_reduction_shape(view_shape.as_slice(), canonical.as_deref(), keepdims)?;

    let count = if let Some(ref canonical) = canonical {
        canonical.iter().map(|&a| view_shape[a]).product::<usize>()
    } else {
        view_shape.iter().product::<usize>()
    };

    if count == 0 {
        return Err(Error::OpError("cannot compute mean of empty tensor".into()));
    }

    let output_numel: usize = if output_shape.is_empty() {
        1
    } else {
        output_shape.iter().product()
    };

    let mut out_storage = allocator.allocate(output_numel)?;
    let out_data = out_storage.try_as_uninit_slice_mut()?;

    for slot in out_data.iter_mut().take(output_numel) {
        slot.write(D::default());
    }

    let view_data = view.storage.as_uninit_slice();

    if axes.is_none() {
        let mut sum = D::default();
        if view.layout.is_contiguous() && view.layout.offset_bytes() == 0 {
            let numel: usize = view_shape.iter().product();
            for slot in view_data.iter().take(numel) {
                sum = sum + unsafe { slot.assume_init() };
            }
        } else {
            for idx in view.layout.iter_element_indices(D::DTYPE)? {
                sum = sum + unsafe { view_data[idx].assume_init() };
            }
        }
        let mean = sum / D::from_usize(count);
        out_data[0].write(mean);
    } else {
        let canonical = canonical.unwrap();

        for (logical_idx, idx) in view.layout.iter_element_indices(D::DTYPE)?.enumerate() {
            let value = unsafe { view_data[idx].assume_init() };

            let input_indices = compute_multi_index_from_linear(logical_idx, view_shape.as_slice());

            let output_linear_idx =
                compute_output_linear_index(&input_indices, &canonical, &output_shape, keepdims);

            let current = unsafe { out_data[output_linear_idx].assume_init_ref() };
            out_data[output_linear_idx].write(*current + value);
        }

        let count_per_output = D::from_usize(count);
        for slot in out_data.iter_mut().take(output_numel) {
            let sum_val = unsafe { slot.assume_init() };
            slot.write(sum_val / count_per_output);
        }
    }

    let out_layout = Layout::contiguous(Shape::from_slice(&output_shape)?);

    Ok(TensorParts {
        storage: out_storage,
        layout: out_layout,
    })
}

impl MeanKernel for f32 {
    fn mean_kernel(
        view: CpuTensorView<'_, Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        reduce_mean(view, axes, keepdims, allocator)
    }
}

impl MeanKernel for f64 {
    fn mean_kernel(
        view: CpuTensorView<'_, Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        reduce_mean(view, axes, keepdims, allocator)
    }
}
