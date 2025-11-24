use crate::{
    dtype::DType,
    error::{Error, Result},
    shape::{ConcreteShape, MAX_RANK},
};
use tinyvec::ArrayVec;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutKind {
    Contiguous,
    General,
}

#[derive(Clone, Debug)]
pub struct Layout {
    shape: ConcreteShape,
    strides: ArrayVec<[isize; MAX_RANK]>,
    offset_bytes: usize,
    kind: LayoutKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorIndexer {
    Select(usize),
    Slice {
        start: usize,
        end: usize,
        step: usize,
    },
}

impl Layout {
    pub fn contiguous(shape: ConcreteShape) -> Self {
        let strides = shape.contiguous_strides();
        Self::new_unchecked(shape, strides, 0)
    }

    pub fn contiguous_with_offset(shape: ConcreteShape, offset_bytes: usize) -> Self {
        let strides = shape.contiguous_strides();
        Self::new_unchecked(shape, strides, offset_bytes)
    }

    pub fn with_strides(
        shape: ConcreteShape,
        strides: &[isize],
        offset_bytes: usize,
    ) -> Result<Self> {
        if shape.rank() != strides.len() {
            return Err(Error::invalid_shape("strides rank must match shape rank"));
        }
        let mut stride_store = ArrayVec::<[isize; MAX_RANK]>::new();
        for &stride in strides {
            stride_store.push(stride);
        }
        Ok(Self::new_unchecked(shape, stride_store, offset_bytes))
    }

    pub fn shape(&self) -> &[usize] {
        self.shape.as_slice()
    }

    pub fn concrete_shape(&self) -> &ConcreteShape {
        &self.shape
    }

    pub fn strides(&self) -> &[isize] {
        self.strides.as_slice()
    }

    pub fn offset_bytes(&self) -> usize {
        self.offset_bytes
    }

    pub fn kind(&self) -> LayoutKind {
        self.kind
    }

    pub fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }

    pub fn is_contiguous(&self) -> bool {
        self.kind == LayoutKind::Contiguous
    }

    pub fn perform_indexing(&self, indexers: &[TensorIndexer], dtype: DType) -> Result<Self> {
        if indexers.len() > self.shape.rank() {
            return Err(Error::invalid_shape(format!(
                "too many indices: expected {}, got {}",
                self.shape.rank(),
                indexers.len()
            )));
        }

        let mut new_shape = Vec::new();
        let mut new_strides = ArrayVec::<[isize; MAX_RANK]>::new();
        let mut current_offset_bytes = self.offset_bytes;
        let elem_size = dtype.size_in_bytes() as isize;

        // Iterate over the layout's dimensions.
        // If an indexer is provided for a dimension, apply it.
        // Otherwise, treat it as a full slice (keep the dimension as-is).
        for (i, (&dim, &stride)) in self
            .shape()
            .iter()
            .zip(self.strides.iter())
            .enumerate()
        {
            let indexer = indexers.get(i).copied().unwrap_or(TensorIndexer::Slice {
                start: 0,
                end: dim,
                step: 1,
            });

            // Calculate the offset adjustment and new dimension (if any)
            let (offset_delta, new_dim_info) = match indexer {
                TensorIndexer::Select(idx) => {
                    if idx >= dim {
                        return Err(Error::invalid_shape(format!(
                            "index {idx} out of bounds for dim {i} (size {dim})"
                        )));
                    }
                    let delta = stride
                        .checked_mul(idx as isize)
                        .ok_or_else(|| Error::invalid_shape("indexer offset overflow"))?;
                    (delta, None)
                }
                TensorIndexer::Slice { start, end, step } => {
                    if step == 0 {
                        return Err(Error::invalid_shape("slice step cannot be zero"));
                    }
                    if start > dim || end > dim {
                        return Err(Error::invalid_shape(format!(
                            "slice range [{start}, {end}) out of bounds for dim {i} (size {dim})"
                        )));
                    }
                    let len = if start >= end {
                        0
                    } else {
                        (end - start).div_ceil(step)
                    };
                    let delta = stride
                        .checked_mul(start as isize)
                        .ok_or_else(|| Error::invalid_shape("slice offset overflow"))?;
                    let new_stride = stride * (step as isize);
                    (delta, Some((len, new_stride)))
                }
            };

            // Apply byte offset delta
            let byte_delta = offset_delta
                .checked_mul(elem_size)
                .ok_or_else(|| Error::invalid_shape("byte offset overflow"))?;

            current_offset_bytes = isize::try_from(current_offset_bytes)
                .map_err(|_| Error::invalid_shape("base offset exceeds isize range"))?
                .checked_add(byte_delta)
                .ok_or_else(|| Error::invalid_shape("total offset overflow"))?
                .try_into()
                .map_err(|_| Error::invalid_shape("total offset negative"))?;

            // Push new dimension info if slicing
            if let Some((len, new_stride)) = new_dim_info {
                new_shape.push(len);
                new_strides.push(new_stride);
            }
        }

        let shape = ConcreteShape::from_slice(&new_shape)?;
        Layout::with_strides(shape, new_strides.as_slice(), current_offset_bytes)
    }

    pub fn slice(
        &self,
        axis: usize,
        start: usize,
        end: usize,
        step: usize,
        dtype: DType,
    ) -> Result<Self> {
        if axis >= self.shape.rank() {
            return Err(Error::InvalidAxes(format!(
                "axis {axis} out of bounds for rank {}",
                self.shape.rank()
            )));
        }
        let mut indexers = Vec::with_capacity(self.shape.rank());
        for i in 0..self.shape.rank() {
            if i == axis {
                indexers.push(TensorIndexer::Slice { start, end, step });
            } else {
                indexers.push(TensorIndexer::Slice {
                    start: 0,
                    end: self.shape()[i],
                    step: 1,
                });
            }
        }
        self.perform_indexing(&indexers, dtype)
    }

    pub fn permute(&self, axes: &[usize]) -> Result<Self> {
        if axes.len() != self.shape.rank() {
            return Err(Error::invalid_shape(
                "permute axes length must match tensor rank",
            ));
        }
        let mut seen = vec![false; axes.len()];
        let rank = self.shape.rank();
        for &axis in axes {
            if axis >= rank {
                return Err(Error::invalid_shape("permute axis out of bounds"));
            }
            if seen[axis] {
                return Err(Error::invalid_shape("duplicate axis in permute"));
            }
            seen[axis] = true;
        }
        let mut new_shape = Vec::with_capacity(axes.len());
        let mut new_strides = ArrayVec::<[isize; MAX_RANK]>::new();
        for &axis in axes {
            new_shape.push(self.shape()[axis]);
            new_strides.push(self.strides()[axis]);
        }
        let shape = ConcreteShape::from_slice(&new_shape)?;
        Layout::with_strides(shape, new_strides.as_slice(), self.offset_bytes)
    }

    pub fn transpose(&self, axis_a: usize, axis_b: usize) -> Result<Self> {
        if axis_a >= self.shape.rank() || axis_b >= self.shape.rank() {
            return Err(Error::invalid_shape("transpose axis out of bounds"));
        }
        let mut axes: Vec<usize> = (0..self.shape.rank()).collect();
        axes.swap(axis_a, axis_b);
        self.permute(&axes)
    }

    pub fn reshape(&self, new_shape: ConcreteShape) -> Result<Self> {
        if new_shape.num_elements() != self.shape.num_elements() {
            return Err(Error::SizeMismatch {
                expected: self.shape.num_elements(),
                actual: new_shape.num_elements(),
            });
        }
        if !self.is_contiguous() {
            return Err(Error::invalid_shape(
                "reshape requires contiguous layout; call contiguous() first",
            ));
        }
        Ok(Layout::contiguous_with_offset(new_shape, self.offset_bytes))
    }

    pub fn offset_elements(&self, dtype: DType) -> isize {
        (self.offset_bytes / dtype.size_in_bytes()) as isize
    }

    pub fn validate_bounds(&self, dtype: DType, buffer_len_bytes: usize) -> Result<()> {
        let end = self.max_offset_bytes(dtype)?;
        if end >= buffer_len_bytes {
            return Err(Error::invalid_shape(format!(
                "layout exceeds buffer bounds: end={} bytes, buffer_len={} bytes",
                end, buffer_len_bytes
            )));
        }
        Ok(())
    }

    pub fn max_offset_bytes(&self, dtype: DType) -> Result<usize> {
        let elem_size = dtype.size_in_bytes();
        let mut max_bytes = isize::try_from(self.offset_bytes)
            .map_err(|_| Error::invalid_shape("layout offset exceeds addressable isize range"))?;
        for (dim, stride) in self.shape.as_slice().iter().zip(self.strides.iter()) {
            if *dim == 0 {
                continue;
            }
            if *stride < 0 {
                return Err(Error::invalid_shape(
                    "negative strides are not supported in bounds check",
                ));
            }
            let extent = stride
                .checked_mul((*dim as isize).saturating_sub(1))
                .ok_or_else(|| Error::invalid_shape("stride*extent overflow"))?;
            let extent_bytes = extent
                .checked_mul(elem_size as isize)
                .ok_or_else(|| Error::invalid_shape("byte offset overflow"))?;
            max_bytes = max_bytes
                .checked_add(extent_bytes)
                .ok_or_else(|| Error::invalid_shape("byte offset overflow"))?;
        }
        let last_byte = max_bytes
            .checked_add((elem_size as isize).saturating_sub(1))
            .ok_or_else(|| Error::invalid_shape("byte offset overflow (last byte)"))?;
        usize::try_from(last_byte)
            .map_err(|_| Error::invalid_shape("byte offset exceeds addressable range"))
    }

    pub fn offset_bytes_for_indices(&self, indices: &[usize], dtype: DType) -> Result<usize> {
        if indices.len() != self.shape.rank() {
            return Err(Error::invalid_shape("index rank must match tensor rank"));
        }
        let elem_size = dtype.size_in_bytes() as isize;
        let mut offset = isize::try_from(self.offset_bytes)
            .map_err(|_| Error::invalid_shape("layout offset exceeds addressable isize range"))?;
        for ((&idx, &dim), &stride) in indices
            .iter()
            .zip(self.shape.as_slice().iter())
            .zip(self.strides.iter())
        {
            if idx >= dim {
                return Err(Error::invalid_shape("index out of bounds"));
            }
            let delta = stride
                .checked_mul(idx as isize)
                .ok_or_else(|| Error::invalid_shape("index stride multiplication overflow"))?;
            let bytes = delta
                .checked_mul(elem_size)
                .ok_or_else(|| Error::invalid_shape("index byte offset overflow"))?;
            offset = offset
                .checked_add(bytes)
                .ok_or_else(|| Error::invalid_shape("index byte offset overflow (accumulated)"))?;
        }
        usize::try_from(offset)
            .map_err(|_| Error::invalid_shape("index offset exceeds addressable range"))
    }

    fn new_unchecked(
        shape: ConcreteShape,
        strides: ArrayVec<[isize; MAX_RANK]>,
        offset_bytes: usize,
    ) -> Self {
        let kind = compute_kind(shape.as_slice(), strides.as_slice());
        Self {
            shape,
            strides,
            offset_bytes,
            kind,
        }
    }
}

fn compute_kind(shape: &[usize], strides: &[isize]) -> LayoutKind {
    let mut expected = 1isize;
    for (dim, stride) in shape.iter().rev().zip(strides.iter().rev()) {
        if *stride != expected {
            return LayoutKind::General;
        }
        expected *= *dim as isize;
    }
    LayoutKind::Contiguous
}
