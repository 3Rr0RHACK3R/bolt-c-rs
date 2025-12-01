use bolt_core::{
    StorageAllocator, TensorParts,
    dtype::NativeType,
    error::{Error, Result},
    layout::Layout,
    shape::{canonical_axes, ConcreteShape},
};

use super::super::allocator::CpuAllocator;
use super::super::storage::{CpuStorage, CpuTensorView};
use super::reduction_helpers::{
    compute_multi_index_from_linear, compute_output_linear_index, compute_reduction_shape,
};

pub trait MinKernel: NativeType {
    fn min_kernel(
        _view: CpuTensorView<'_, Self>,
        _axes: Option<&[usize]>,
        _keepdims: bool,
        _allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "min not implemented for {}",
            Self::DTYPE
        )))
    }
}

fn reduce_min<D>(
    input: CpuTensorView<'_, D>,
    axes: Option<&[usize]>,
    keepdims: bool,
    allocator: &CpuAllocator<D>,
) -> Result<TensorParts<CpuStorage<D>>>
where
    D: NativeType + Copy + PartialOrd,
{
    let input_shape = input.layout.shape();
    let output_shape = compute_reduction_shape(input_shape, axes, keepdims)?;

    if let Some(ax) = axes {
        canonical_axes(ax, input_shape.len())?;
    }

    let output_numel: usize = if output_shape.is_empty() {
        1
    } else {
        output_shape.iter().product()
    };

    let mut out_storage = allocator.allocate(output_numel)?;
    let out_data = out_storage.try_as_uninit_slice_mut()?;

    let input_data = input.storage.as_uninit_slice();
    let elem_size = D::DTYPE.size_in_bytes();

    if axes.is_none() {
        let mut min_val = None;
        if input.layout.is_contiguous() && input.layout.offset_bytes() == 0 {
            let numel: usize = input_shape.iter().product();
            for slot in input_data.iter().take(numel) {
                let val = unsafe { slot.assume_init() };
                min_val = Some(match min_val {
                    None => val,
                    Some(current) => if val < current { val } else { current },
                });
            }
        } else {
            for byte_offset in input.layout.iter_offsets(D::DTYPE)? {
                let idx = byte_offset / elem_size;
                let val = unsafe { input_data[idx].assume_init() };
                min_val = Some(match min_val {
                    None => val,
                    Some(current) => if val < current { val } else { current },
                });
            }
        }
        out_data[0].write(min_val.ok_or_else(|| Error::OpError("cannot compute min of empty tensor".into()))?);
    } else {
        let canonical = canonical_axes(axes.unwrap(), input_shape.len())?;

        let mut initialized = vec![false; output_numel];

        for byte_offset in input.layout.iter_offsets(D::DTYPE)? {
            let idx = byte_offset / elem_size;
            let value = unsafe { input_data[idx].assume_init() };

            let input_linear_idx = if input.layout.is_contiguous()
                && input.layout.offset_bytes() == 0
            {
                idx
            } else {
                (byte_offset - input.layout.offset_bytes()) / elem_size
            };

            let input_indices = compute_multi_index_from_linear(input_linear_idx, input_shape);

            let output_linear_idx =
                compute_output_linear_index(&input_indices, &canonical, &output_shape, keepdims);

            if !initialized[output_linear_idx] {
                out_data[output_linear_idx].write(value);
                initialized[output_linear_idx] = true;
            } else {
                let current = unsafe { out_data[output_linear_idx].assume_init_ref() };
                if value < *current {
                    out_data[output_linear_idx].write(value);
                }
            }
        }
    }

    let out_layout = Layout::contiguous(ConcreteShape::from_slice(&output_shape)?);

    Ok(TensorParts {
        storage: out_storage,
        layout: out_layout,
    })
}

impl MinKernel for f32 {
    fn min_kernel(
        view: CpuTensorView<'_, Self>,
        axes: Option<&[usize]>,
        keepdims: bool,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        reduce_min(view, axes, keepdims, allocator)
    }
}

impl MinKernel for f64 {
    fn min_kernel(
        view: CpuTensorView<'_, Self>,
        axes: Option<&[usize]>,
        keepdims: bool,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        reduce_min(view, axes, keepdims, allocator)
    }
}

impl MinKernel for i32 {
    fn min_kernel(
        view: CpuTensorView<'_, Self>,
        axes: Option<&[usize]>,
        keepdims: bool,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        reduce_min(view, axes, keepdims, allocator)
    }
}
