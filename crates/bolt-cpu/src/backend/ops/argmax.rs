use bolt_core::{
    StorageAllocator, TensorParts,
    dtype::NativeType,
    error::{Error, Result},
    layout::Layout,
    shape::{ConcreteShape, canonical_axes},
};

use super::super::allocator::CpuAllocator;
use super::super::storage::{CpuStorage, CpuTensorView};
use super::reduction_helpers::{
    compute_multi_index_from_linear, compute_output_linear_index, compute_reduction_shape,
};

pub trait ArgmaxKernel: NativeType {
    fn argmax_kernel(
        _view: CpuTensorView<'_, Self>,
        _axes: Option<&[isize]>,
        _keepdims: bool,
        _allocator: &CpuAllocator<i32>,
    ) -> Result<TensorParts<CpuStorage<i32>>> {
        Err(Error::OpError(format!(
            "argmax not implemented for {}",
            Self::DTYPE
        )))
    }
}

fn reduce_argmax<D>(
    view: CpuTensorView<'_, D>,
    axes: Option<&[isize]>,
    keepdims: bool,
    allocator: &CpuAllocator<i32>,
) -> Result<TensorParts<CpuStorage<i32>>>
where
    D: NativeType + Copy + PartialOrd,
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

    let view_data = view.storage.as_uninit_slice();

    if axes.is_none() {
        let mut max_val = None;
        let mut max_idx = 0i32;
        let mut curr_idx = 0i32;

        if view.layout.is_contiguous() && view.layout.offset_bytes() == 0 {
            let numel: usize = view_shape.iter().product();
            for slot in view_data.iter().take(numel) {
                let val = unsafe { slot.assume_init() };
                match max_val {
                    None => {
                        max_val = Some(val);
                        max_idx = curr_idx;
                    }
                    Some(current) => {
                        if val > current {
                            max_val = Some(val);
                            max_idx = curr_idx;
                        }
                    }
                }
                curr_idx += 1;
            }
        } else {
            for idx in view.layout.iter_element_indices(D::DTYPE)? {
                let val = unsafe { view_data[idx].assume_init() };
                match max_val {
                    None => {
                        max_val = Some(val);
                        max_idx = curr_idx;
                    }
                    Some(current) => {
                        if val > current {
                            max_val = Some(val);
                            max_idx = curr_idx;
                        }
                    }
                }
                curr_idx += 1;
            }
        }

        if max_val.is_none() {
            return Err(Error::OpError(
                "cannot compute argmax of empty tensor".into(),
            ));
        }
        out_data[0].write(max_idx);
    } else {
        let canonical = canonical.unwrap();

        let mut max_vals: Vec<Option<D>> = vec![None; output_numel];
        let mut max_indices: Vec<i32> = vec![0; output_numel];

        for (logical_idx, idx) in view
            .layout
            .iter_element_indices(D::DTYPE)?
            .enumerate()
        {
            let value = unsafe { view_data[idx].assume_init() };

            let input_indices = compute_multi_index_from_linear(logical_idx, view_shape);

            let output_linear_idx =
                compute_output_linear_index(&input_indices, &canonical, &output_shape, keepdims);

            let axis_index = input_indices[canonical[0]] as i32;

            match max_vals[output_linear_idx] {
                None => {
                    max_vals[output_linear_idx] = Some(value);
                    max_indices[output_linear_idx] = axis_index;
                }
                Some(current) => {
                    if value > current {
                        max_vals[output_linear_idx] = Some(value);
                        max_indices[output_linear_idx] = axis_index;
                    }
                }
            }
            
        }

        for (i, idx_val) in max_indices.iter().enumerate() {
            out_data[i].write(*idx_val);
        }
    }

    let out_layout = Layout::contiguous(ConcreteShape::from_slice(&output_shape)?);

    Ok(TensorParts {
        storage: out_storage,
        layout: out_layout,
    })
}

impl ArgmaxKernel for f32 {
    fn argmax_kernel(
        view: CpuTensorView<'_, Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        allocator: &CpuAllocator<i32>,
    ) -> Result<TensorParts<CpuStorage<i32>>> {
        reduce_argmax(view, axes, keepdims, allocator)
    }
}

impl ArgmaxKernel for f64 {
    fn argmax_kernel(
        view: CpuTensorView<'_, Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        allocator: &CpuAllocator<i32>,
    ) -> Result<TensorParts<CpuStorage<i32>>> {
        reduce_argmax(view, axes, keepdims, allocator)
    }
}

impl ArgmaxKernel for i32 {
    fn argmax_kernel(
        view: CpuTensorView<'_, Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        allocator: &CpuAllocator<i32>,
    ) -> Result<TensorParts<CpuStorage<i32>>> {
        reduce_argmax(view, axes, keepdims, allocator)
    }
}
