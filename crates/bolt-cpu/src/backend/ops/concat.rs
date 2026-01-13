use bolt_core::{
    StorageAllocator, TensorParts,
    dtype::NativeType,
    error::{Error, Result},
    layout::Layout,
    shape::{MAX_RANK, Shape},
};

use super::super::allocator::CpuAllocator;
use super::super::storage::{CpuStorage, CpuTensorView};

pub trait ConcatKernel: NativeType {
    fn concat_kernel(
        _tensors: &[(CpuTensorView<'_, Self>, usize)],
        _axis: usize,
        _output_shape: &Shape,
        _allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        Err(Error::OpError(format!(
            "concat not implemented for {}",
            Self::DTYPE
        )))
    }
}

fn product(dims: &[usize]) -> usize {
    dims.iter().copied().product::<usize>()
}

/// Allocation-free conversion of a linear index into a multi-index (row-major).
/// Writes into `out[..shape.len()]`.
fn linear_to_multi_index_inplace(mut linear: usize, shape: &[usize], out: &mut [usize]) {
    debug_assert!(out.len() >= shape.len());
    for (i, &dim) in shape.iter().enumerate().rev() {
        out[i] = linear % dim;
        linear /= dim;
    }
}

fn multi_to_linear_index_strided(indices: &[usize], strides: &[isize]) -> usize {
    debug_assert_eq!(indices.len(), strides.len());
    let mut linear = 0usize;
    for (&idx, &stride) in indices.iter().zip(strides.iter()) {
        // Strides should be positive for contiguous row-major outputs.
        linear += idx * (stride as usize);
    }
    linear
}

fn concat_contiguous<D>(
    tensors: &[(CpuTensorView<'_, D>, usize)],
    axis: usize,
    output_shape: &Shape,
    allocator: &CpuAllocator<D>,
) -> Result<TensorParts<CpuStorage<D>>>
where
    D: NativeType + Copy,
{
    // Preconditions: all inputs are contiguous and have offset 0; validated by caller.
    let output_dims = output_shape.as_slice();
    let rank = output_dims.len();
    debug_assert!(axis < rank);

    let numel = output_shape.num_elements();
    let mut out_storage: CpuStorage<D> = allocator.allocate(numel)?;
    let out_slice = out_storage.try_as_uninit_slice_mut()?;

    let outer = product(&output_dims[..axis]);
    let inner = product(&output_dims[axis + 1..]);
    let out_axis = output_dims[axis];

    // For each outer index, copy each tensor's (axis_size * inner) block into the right spot.
    for outer_idx in 0..outer.max(1) {
        let mut axis_offset = 0usize;
        for (view, axis_size) in tensors {
            let in_shape = view.layout.shape().as_slice();
            let in_axis = in_shape[axis];
            debug_assert_eq!(in_axis, *axis_size);

            let block_len = in_axis * inner;
            let src_base = outer_idx * in_axis * inner;
            let dst_base = outer_idx * out_axis * inner + axis_offset * inner;

            let src = &view.storage.as_uninit_slice()[src_base..src_base + block_len];
            let dst = &mut out_slice[dst_base..dst_base + block_len];
            dst.copy_from_slice(src);

            axis_offset += in_axis;
        }
    }

    let layout = Layout::contiguous(output_shape.clone());
    Ok(TensorParts {
        storage: out_storage,
        layout,
    })
}

pub fn concat<D>(
    tensors: &[(CpuTensorView<'_, D>, usize)],
    axis: usize,
    output_shape: &Shape,
    allocator: &CpuAllocator<D>,
) -> Result<TensorParts<CpuStorage<D>>>
where
    D: NativeType + Copy,
{
    if tensors.is_empty() {
        return Err(Error::invalid_shape("cannot concat empty tensor list"));
    }

    let rank = output_shape.rank();
    if axis >= rank {
        return Err(Error::invalid_shape(format!(
            "axis {} out of bounds for rank {}",
            axis, rank
        )));
    }

    // Fast path: all inputs contiguous + offset 0 => block copies (no per-element math)
    let all_contiguous = tensors.iter().all(|(t, _)| t.layout.is_contiguous() && t.layout.offset_bytes() == 0);
    if all_contiguous {
        return concat_contiguous(tensors, axis, output_shape, allocator);
    }

    // General path: avoid heap allocs per element by using a fixed-size index buffer.
    let numel = output_shape.num_elements();
    let mut out_storage: CpuStorage<D> = allocator.allocate(numel)?;
    let out_slice = out_storage.try_as_uninit_slice_mut()?;
    let output_strides = output_shape.contiguous_strides();

    let mut axis_cumsum = 0usize;
    for (tensor_view, axis_size) in tensors {
        let input_data = tensor_view.storage.as_uninit_slice();
        let input_layout = tensor_view.layout;
        let input_shape = input_layout.shape();
        let input_shape_slice = input_shape.as_slice();

        // Shape::MAX_RANK is 12; keep this on the stack.
        debug_assert!(input_shape_slice.len() <= MAX_RANK);
        let mut idx_buf = [0usize; MAX_RANK];

        for (logical_idx, physical_idx) in input_layout.iter_element_indices(D::DTYPE)?.enumerate() {
            linear_to_multi_index_inplace(logical_idx, input_shape_slice, &mut idx_buf);
            idx_buf[axis] += axis_cumsum;
            let out_idx = multi_to_linear_index_strided(&idx_buf[..rank], output_strides.as_slice());
            let value = unsafe { input_data[physical_idx].assume_init() };
            out_slice[out_idx].write(value);
            idx_buf[axis] -= axis_cumsum;
        }

        axis_cumsum += axis_size;
    }

    let layout = Layout::contiguous(output_shape.clone());
    Ok(TensorParts { storage: out_storage, layout })
}

impl ConcatKernel for f32 {
    fn concat_kernel(
        tensors: &[(CpuTensorView<'_, Self>, usize)],
        axis: usize,
        output_shape: &Shape,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        concat(tensors, axis, output_shape, allocator)
    }
}

impl ConcatKernel for f64 {
    fn concat_kernel(
        tensors: &[(CpuTensorView<'_, Self>, usize)],
        axis: usize,
        output_shape: &Shape,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        concat(tensors, axis, output_shape, allocator)
    }
}

impl ConcatKernel for i32 {
    fn concat_kernel(
        tensors: &[(CpuTensorView<'_, Self>, usize)],
        axis: usize,
        output_shape: &Shape,
        allocator: &CpuAllocator<Self>,
    ) -> Result<TensorParts<CpuStorage<Self>>> {
        concat(tensors, axis, output_shape, allocator)
    }
}
