use bolt_core::{error::Result, shape::reduced_shape};

pub(super) fn compute_reduction_shape(
    shape: &[usize],
    canonical_axes: Option<&[usize]>,
    keepdims: bool,
) -> Result<Vec<usize>> {
    match canonical_axes {
        None => {
            if keepdims {
                Ok(vec![1; shape.len()])
            } else {
                Ok(vec![])
            }
        }
        Some(canonical) => {
            if keepdims {
                let mut result = shape.to_vec();
                for &axis in canonical {
                    result[axis] = 1;
                }
                Ok(result)
            } else {
                reduced_shape(
                    shape,
                    &canonical.iter().map(|&x| x as isize).collect::<Vec<_>>(),
                )
            }
        }
    }
}

pub(super) fn compute_multi_index_from_linear(
    mut linear_idx: usize,
    shape: &[usize],
) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    for i in (0..shape.len()).rev() {
        indices[i] = linear_idx % shape[i];
        linear_idx /= shape[i];
    }
    indices
}

pub(super) fn compute_output_linear_index(
    input_indices: &[usize],
    reduced_axes: &[usize],
    output_shape: &[usize],
    keepdims: bool,
) -> usize {
    let mut linear = 0;
    let mut stride = 1;

    if keepdims {
        for i in (0..input_indices.len()).rev() {
            if !reduced_axes.contains(&i) {
                linear += input_indices[i] * stride;
            }
            stride *= output_shape[i];
        }
    } else {
        let mut out_axis = output_shape.len();
        for i in (0..input_indices.len()).rev() {
            if !reduced_axes.contains(&i) {
                out_axis -= 1;
                linear += input_indices[i] * stride;
                if out_axis > 0 {
                    stride *= output_shape[out_axis - 1];
                }
            }
        }
    }

    linear
}
