use bolt_core::{
    error::{Error, Result},
    layout::Layout,
};

pub fn linear_to_indices(index: usize, shape: &[usize], coords: &mut [usize]) {
    let mut value = index;
    for (axis, &dim) in shape.iter().enumerate().rev() {
        coords[axis] = value % dim;
        value /= dim;
    }
}

pub fn offset_from_strides(indices: &[usize], strides: &[isize]) -> isize {
    indices
        .iter()
        .zip(strides.iter())
        .map(|(&idx, &stride)| idx as isize * stride)
        .sum()
}

pub fn expand_strides(layout: &Layout, target_shape: &[usize]) -> Result<Vec<isize>> {
    let in_shape = layout.shape();
    let in_strides = layout.strides();
    let mut result = vec![0isize; target_shape.len()];
    let mut in_idx = in_shape.len();
    for out_idx in (0..target_shape.len()).rev() {
        let out_dim = target_shape[out_idx];
        if in_idx > 0 {
            let current_in = in_idx - 1;
            let in_dim = in_shape[current_in];
            if in_dim == out_dim {
                result[out_idx] = in_strides[current_in];
                in_idx -= 1;
            } else if in_dim == 1 {
                result[out_idx] = 0;
                in_idx -= 1;
            } else {
                return Err(Error::ShapeMismatch {
                    lhs: in_shape.to_vec(),
                    rhs: target_shape.to_vec(),
                });
            }
        } else {
            result[out_idx] = 0;
        }
    }
    Ok(result)
}
