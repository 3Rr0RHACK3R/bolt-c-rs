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

pub trait ArgminKernel: NativeType {
    fn argmin_kernel(
        _view: CpuTensorView<'_, Self>,
        _axes: Option<&[usize]>,
        _keepdims: bool,
        _allocator: &CpuAllocator<i32>,
    ) -> Result<TensorParts<CpuStorage<i32>>> {
        Err(Error::OpError(format!(
            "argmin not implemented for {}",
            Self::DTYPE
        )))
    }
}

fn reduce_argmin<D>(
    input: CpuTensorView<'_, D>,
    axes: Option<&[usize]>,
    keepdims: bool,
    allocator: &CpuAllocator<i32>,
) -> Result<TensorParts<CpuStorage<i32>>>
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
        let mut min_idx = 0i32;
        let mut curr_idx = 0i32;

        if input.layout.is_contiguous() && input.layout.offset_bytes() == 0 {
            let numel: usize = input_shape.iter().product();
            for slot in input_data.iter().take(numel) {
                let val = unsafe { slot.assume_init() };
                match min_val {
                    None => {
                        min_val = Some(val);
                        min_idx = curr_idx;
                    }
                    Some(current) => {
                        if val < current {
                            min_val = Some(val);
                            min_idx = curr_idx;
                        }
                    }
                }
                curr_idx += 1;
            }
        } else {
            for byte_offset in input.layout.iter_offsets(D::DTYPE)? {
                let idx = byte_offset / elem_size;
                let val = unsafe { input_data[idx].assume_init() };
                match min_val {
                    None => {
                        min_val = Some(val);
                        min_idx = curr_idx;
                    }
                    Some(current) => {
                        if val < current {
                            min_val = Some(val);
                            min_idx = curr_idx;
                        }
                    }
                }
                curr_idx += 1;
            }
        }

        if min_val.is_none() {
            return Err(Error::OpError("cannot compute argmin of empty tensor".into()));
        }
        out_data[0].write(min_idx);
    } else {
        let canonical = canonical_axes(axes.unwrap(), input_shape.len())?;

        let mut min_vals: Vec<Option<D>> = vec![None; output_numel];
        let mut min_indices: Vec<i32> = vec![0; output_numel];

        let mut logical_idx = 0;
        for byte_offset in input.layout.iter_offsets(D::DTYPE)? {
            let idx = byte_offset / elem_size;
            let value = unsafe { input_data[idx].assume_init() };

            let input_indices = compute_multi_index_from_linear(logical_idx, input_shape);

            let output_linear_idx =
                compute_output_linear_index(&input_indices, &canonical, &output_shape, keepdims);

            let axis_index = input_indices[canonical[0]] as i32;

            match min_vals[output_linear_idx] {
                None => {
                    min_vals[output_linear_idx] = Some(value);
                    min_indices[output_linear_idx] = axis_index;
                }
                Some(current) => {
                    if value < current {
                        min_vals[output_linear_idx] = Some(value);
                        min_indices[output_linear_idx] = axis_index;
                    }
                }
            }

            logical_idx += 1;
        }

        for (i, idx_val) in min_indices.iter().enumerate() {
            out_data[i].write(*idx_val);
        }
    }

    let out_layout = Layout::contiguous(ConcreteShape::from_slice(&output_shape)?);

    Ok(TensorParts {
        storage: out_storage,
        layout: out_layout,
    })
}

impl ArgminKernel for f32 {
    fn argmin_kernel(
        view: CpuTensorView<'_, Self>,
        axes: Option<&[usize]>,
        keepdims: bool,
        allocator: &CpuAllocator<i32>,
    ) -> Result<TensorParts<CpuStorage<i32>>> {
        reduce_argmin(view, axes, keepdims, allocator)
    }
}

impl ArgminKernel for f64 {
    fn argmin_kernel(
        view: CpuTensorView<'_, Self>,
        axes: Option<&[usize]>,
        keepdims: bool,
        allocator: &CpuAllocator<i32>,
    ) -> Result<TensorParts<CpuStorage<i32>>> {
        reduce_argmin(view, axes, keepdims, allocator)
    }
}

impl ArgminKernel for i32 {
    fn argmin_kernel(
        view: CpuTensorView<'_, Self>,
        axes: Option<&[usize]>,
        keepdims: bool,
        allocator: &CpuAllocator<i32>,
    ) -> Result<TensorParts<CpuStorage<i32>>> {
        reduce_argmin(view, axes, keepdims, allocator)
    }
}
