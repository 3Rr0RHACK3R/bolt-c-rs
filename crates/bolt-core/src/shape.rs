use crate::error::{Error, Result};

/// Runtime-validated shape metadata (rank >= 1, every dimension > 0).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConcreteShape {
    dims: Vec<usize>,
}

impl ConcreteShape {
    pub fn from_slice(dims: &[usize]) -> Result<Self> {
        Self::validate_dims(dims)?;
        Ok(Self {
            dims: dims.to_vec(),
        })
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.dims
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn contiguous_strides(&self) -> Vec<isize> {
        let mut strides = vec![0isize; self.rank()];
        let mut stride = 1isize;
        for (i, dim) in self.dims.iter().enumerate().rev() {
            strides[i] = stride;
            stride *= *dim as isize;
        }
        strides
    }

    fn validate_dims(dims: &[usize]) -> Result<()> {
        if dims.is_empty() {
            return Err(Error::invalid_shape(
                "shape must have at least one dimension",
            ));
        }
        if dims.contains(&0) {
            return Err(Error::invalid_shape(
                "zero-sized dimensions are not supported",
            ));
        }
        Ok(())
    }
}

impl From<&ConcreteShape> for ConcreteShape {
    fn from(value: &ConcreteShape) -> Self {
        value.clone()
    }
}

impl TryFrom<Vec<usize>> for ConcreteShape {
    type Error = Error;

    fn try_from(dims: Vec<usize>) -> Result<Self> {
        ConcreteShape::validate_dims(&dims)?;
        Ok(Self { dims })
    }
}

impl TryFrom<&[usize]> for ConcreteShape {
    type Error = Error;

    fn try_from(dims: &[usize]) -> Result<Self> {
        ConcreteShape::from_slice(dims)
    }
}

pub fn broadcast_shapes(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>> {
    let mut shape = Vec::with_capacity(lhs.len().max(rhs.len()));
    let mut l_iter = lhs.iter().rev();
    let mut r_iter = rhs.iter().rev();
    loop {
        let l = l_iter.next();
        let r = r_iter.next();
        match (l, r) {
            (None, None) => break,
            (Some(&lv), None) | (None, Some(&lv)) => {
                shape.push(lv);
            }
            (Some(&lv), Some(&rv)) => {
                if lv == rv {
                    shape.push(lv);
                } else if lv == 1 {
                    shape.push(rv);
                } else if rv == 1 {
                    shape.push(lv);
                } else {
                    return Err(Error::ShapeMismatch {
                        lhs: lhs.to_vec(),
                        rhs: rhs.to_vec(),
                    });
                }
            }
        }
    }
    shape.reverse();
    if shape.is_empty() {
        return Err(Error::invalid_shape(
            "broadcast result cannot be scalar-less",
        ));
    }
    Ok(shape)
}

pub fn canonical_axes(axes: &[usize], rank: usize) -> Result<Vec<usize>> {
    if rank == 0 {
        return Err(Error::invalid_shape("tensor rank must be > 0"));
    }
    if axes.is_empty() {
        return Ok((0..rank).collect());
    }
    let mut seen = vec![false; rank];
    for &axis in axes {
        if axis >= rank {
            return Err(Error::InvalidAxes(format!(
                "axis {axis} out of bounds for rank {rank}"
            )));
        }
        if seen[axis] {
            return Err(Error::InvalidAxes(format!("axis {axis} provided twice")));
        }
        seen[axis] = true;
    }
    let mut out = axes.to_vec();
    out.sort_unstable();
    Ok(out)
}

pub fn reduced_shape(shape: &[usize], axes: &[usize]) -> Result<Vec<usize>> {
    let canonical = canonical_axes(axes, shape.len())?;
    let mut result = Vec::with_capacity(shape.len().saturating_sub(canonical.len()).max(1));
    for (idx, dim) in shape.iter().enumerate() {
        if canonical.binary_search(&idx).is_err() {
            result.push(*dim);
        }
    }
    if result.is_empty() {
        result.push(1);
    }
    Ok(result)
}
