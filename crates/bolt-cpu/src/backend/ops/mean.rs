use bolt_core::{
    StorageAllocator, TensorParts,
    dtype::FloatType,
    error::{Error, Result},
    layout::Layout,
    shape::{ConcreteShape, canonical_axes},
};

use super::super::allocator::CpuAllocator;
use super::super::storage::{CpuStorage, CpuTensorView};
use super::reduction_helpers::{
    compute_multi_index_from_linear, compute_output_linear_index, compute_reduction_shape,
};

pub trait MeanKernel: FloatType {
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
    input: CpuTensorView<'_, D>,
    axes: Option<&[isize]>,
    keepdims: bool,
    allocator: &CpuAllocator<D>,
) -> Result<TensorParts<CpuStorage<D>>>
where
    D: FloatType + num_traits::Float,
{
    let input_shape = input.layout.shape();
    let output_shape = compute_reduction_shape(input_shape, axes, keepdims)?;

    if let Some(ax) = axes {
        canonical_axes(ax, input_shape.len())?;
    }

    // Compute count of elements being reduced
    let count = if let Some(ax) = axes {
        let canonical = canonical_axes(ax, input_shape.len())?;
        canonical.iter().map(|&a| input_shape[a]).product::<usize>()
    } else {
        input_shape.iter().product::<usize>()
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

    let input_data = input.storage.as_uninit_slice();
    let elem_size = D::DTYPE.size_in_bytes();

    if axes.is_none() {
        let mut sum = D::default();
        if input.layout.is_contiguous() && input.layout.offset_bytes() == 0 {
            let numel: usize = input_shape.iter().product();
            for slot in input_data.iter().take(numel) {
                sum = sum + unsafe { slot.assume_init() };
            }
        } else {
            for byte_offset in input.layout.iter_offsets(D::DTYPE)? {
                let idx = byte_offset / elem_size;
                sum = sum + unsafe { input_data[idx].assume_init() };
            }
        }
        let mean = sum / D::from(count).unwrap();
        out_data[0].write(mean);
    } else {
        let canonical = canonical_axes(axes.unwrap(), input_shape.len())?;

        let mut logical_idx = 0;
        for byte_offset in input.layout.iter_offsets(D::DTYPE)? {
            let idx = byte_offset / elem_size;
            let value = unsafe { input_data[idx].assume_init() };

            let input_indices = compute_multi_index_from_linear(logical_idx, input_shape);

            let output_linear_idx =
                compute_output_linear_index(&input_indices, &canonical, &output_shape, keepdims);

            let current = unsafe { out_data[output_linear_idx].assume_init_ref() };
            out_data[output_linear_idx].write(*current + value);

            logical_idx += 1;
        }

        // Divide all accumulated sums by count to get means
        let count_per_output = D::from(count).unwrap();
        for slot in out_data.iter_mut().take(output_numel) {
            let sum_val = unsafe { slot.assume_init() };
            slot.write(sum_val / count_per_output);
        }
    }

    let out_layout = Layout::contiguous(ConcreteShape::from_slice(&output_shape)?);

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
