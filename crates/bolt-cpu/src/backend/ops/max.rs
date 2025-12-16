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

pub trait MaxKernel: NativeType {
    fn max_kernel(
        _view: CpuTensorView<'_, Self>,
        _axes: Option<&[isize]>,
        _keepdims: bool,
        _allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "max not implemented for {}",
            Self::DTYPE
        )))
    }
}

fn reduce_max<D>(
    view: CpuTensorView<'_, D>,
    axes: Option<&[isize]>,
    keepdims: bool,
    allocator: &CpuAllocator<D>,
) -> Result<TensorParts<CpuStorage<D>>>
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
        if view.layout.is_contiguous() && view.layout.offset_bytes() == 0 {
            let numel: usize = view_shape.iter().product();
            for slot in view_data.iter().take(numel) {
                let val = unsafe { slot.assume_init() };
                max_val = Some(match max_val {
                    None => val,
                    Some(current) => {
                        if val > current {
                            val
                        } else {
                            current
                        }
                    }
                });
            }
        } else {
            for idx in view.layout.iter_element_indices(D::DTYPE)? {
                let val = unsafe { view_data[idx].assume_init() };
                max_val = Some(match max_val {
                    None => val,
                    Some(current) => {
                        if val > current {
                            val
                        } else {
                            current
                        }
                    }
                });
            }
        }
        out_data[0].write(
            max_val.ok_or_else(|| Error::OpError("cannot compute max of empty tensor".into()))?,
        );
    } else {
        let canonical = canonical.unwrap();

        let mut initialized = vec![false; output_numel];

        for (logical_idx, idx) in view
            .layout
            .iter_element_indices(D::DTYPE)?
            .enumerate()
        {
            let value = unsafe { view_data[idx].assume_init() };

            let input_indices = compute_multi_index_from_linear(logical_idx, view_shape);

            let output_linear_idx =
                compute_output_linear_index(&input_indices, &canonical, &output_shape, keepdims);

            if !initialized[output_linear_idx] {
                out_data[output_linear_idx].write(value);
                initialized[output_linear_idx] = true;
            } else {
                let current = unsafe { out_data[output_linear_idx].assume_init_ref() };
                if value > *current {
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

impl MaxKernel for f32 {
    fn max_kernel(
        view: CpuTensorView<'_, Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        reduce_max(view, axes, keepdims, allocator)
    }
}

impl MaxKernel for f64 {
    fn max_kernel(
        view: CpuTensorView<'_, Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        reduce_max(view, axes, keepdims, allocator)
    }
}

impl MaxKernel for i32 {
    fn max_kernel(
        view: CpuTensorView<'_, Self>,
        axes: Option<&[isize]>,
        keepdims: bool,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        reduce_max(view, axes, keepdims, allocator)
    }
}
