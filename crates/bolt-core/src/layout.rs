use crate::{
    dtype::DType,
    error::{Error, Result},
    shape::{Shape, MAX_RANK, broadcast_shapes},
};
use tinyvec::ArrayVec;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutKind {
    Contiguous,
    General,
}

#[derive(Clone, Debug)]
pub struct Layout {
    shape: Shape,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IterMode {
    Read,
    Write,
}

#[derive(Clone, Debug)]
pub struct LayoutIter {
    shape: ArrayVec<[usize; MAX_RANK]>,
    indices: ArrayVec<[usize; MAX_RANK]>,
    byte_strides: ArrayVec<[isize; MAX_RANK]>,
    rewinds: ArrayVec<[isize; MAX_RANK]>,
    current_offset: isize,
    remaining_elements: usize,
}

impl Iterator for LayoutIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining_elements == 0 {
            return None;
        }

        let offset = usize::try_from(self.current_offset).expect("offset became negative");
        self.remaining_elements -= 1;

        if self.remaining_elements > 0 {
            for i in (0..self.shape.len()).rev() {
                if self.indices[i] + 1 < self.shape[i] {
                    self.indices[i] += 1;
                    self.current_offset += self.byte_strides[i];
                    break;
                } else {
                    self.indices[i] = 0;
                    self.current_offset -= self.rewinds[i];
                }
            }
        }

        Some(offset)
    }
}

impl Layout {
    pub fn contiguous(shape: Shape) -> Self {
        let strides = shape.contiguous_strides();
        Self::new_unchecked(shape, strides, 0)
    }

    pub fn contiguous_with_offset(shape: Shape, offset_bytes: usize) -> Self {
        let strides = shape.contiguous_strides();
        Self::new_unchecked(shape, strides, offset_bytes)
    }

    pub fn with_strides(
        shape: Shape,
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

    pub fn shape(&self) -> &Shape {
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
        for (i, (&dim, &stride)) in self.shape().iter().zip(self.strides.iter()).enumerate() {
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
                            "index {idx} out of bounds for axis {i} (size {dim})"
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
                            "slice range [{start}, {end}) out of bounds for axis {i} (size {dim})"
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
                    let step_isize = isize::try_from(step)
                        .map_err(|_| Error::invalid_shape("slice step exceeds isize range"))?;
                    let new_stride = stride
                        .checked_mul(step_isize)
                        .ok_or_else(|| Error::invalid_shape("slice stride overflow"))?;
                    (delta, Some((len, new_stride)))
                }
            };

            // Apply byte offset delta
            let byte_delta = offset_delta
                .checked_mul(elem_size)
                .ok_or_else(|| Error::invalid_shape("byte offset overflow"))?;

            let base_offset_isize = isize::try_from(current_offset_bytes)
                .map_err(|_| Error::invalid_shape("base offset exceeds isize range"))?;
            let updated_offset = base_offset_isize
                .checked_add(byte_delta)
                .ok_or_else(|| Error::invalid_shape("total offset overflow"))?;
            current_offset_bytes = updated_offset
                .try_into()
                .map_err(|_| Error::invalid_shape("total offset negative"))?;

            // Push new dimension info if slicing
            if let Some((len, new_stride)) = new_dim_info {
                new_shape.push(len);
                new_strides.push(new_stride);
            }
        }

        let shape = Shape::from_slice(&new_shape)?;
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

    pub fn permute(&self, axes: &[isize]) -> Result<Self> {
        let rank = self.shape.rank();
        if axes.len() != rank {
            return Err(Error::invalid_shape(
                "permute axes length must match tensor rank",
            ));
        }

        let mut normalized = Vec::with_capacity(axes.len());
        for &axis in axes {
            normalized.push(crate::shape::normalize_axis(axis, rank)?);
        }

        let mut seen = vec![false; rank];
        for &axis in &normalized {
            if seen[axis] {
                return Err(Error::invalid_shape("duplicate axis in permute"));
            }
            seen[axis] = true;
        }

        let mut new_shape = Vec::with_capacity(axes.len());
        let mut new_strides = ArrayVec::<[isize; MAX_RANK]>::new();
        for &axis in &normalized {
            new_shape.push(self.shape()[axis]);
            new_strides.push(self.strides()[axis]);
        }
        let shape = Shape::from_slice(&new_shape)?;
        Layout::with_strides(shape, new_strides.as_slice(), self.offset_bytes)
    }

    pub fn transpose(&self, axis_a: isize, axis_b: isize) -> Result<Self> {
        let rank = self.shape.rank();
        let norm_a = crate::shape::normalize_axis(axis_a, rank)?;
        let norm_b = crate::shape::normalize_axis(axis_b, rank)?;
        let mut axes: Vec<isize> = (0..rank).map(|i| i as isize).collect();
        axes.swap(norm_a, norm_b);
        self.permute(&axes)
    }

    pub fn reshape(&self, new_shape: Shape) -> Result<Self> {
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

    pub fn squeeze_all(&self) -> Result<Self> {
        if self.shape.rank() == 0 {
            return Ok(self.clone());
        }
        let mut new_shape = Vec::new();
        let mut new_strides = ArrayVec::<[isize; MAX_RANK]>::new();
        for (&dim, &stride) in self.shape().iter().zip(self.strides().iter()) {
            if dim != 1 {
                new_shape.push(dim);
                new_strides.push(stride);
            }
        }
        if new_shape.len() == self.shape.rank() {
            return Ok(self.clone());
        }
        let shape = Shape::from_slice(&new_shape)?;
        Layout::with_strides(shape, new_strides.as_slice(), self.offset_bytes)
    }

    pub fn squeeze_axis(&self, axis: isize) -> Result<Self> {
        let rank = self.shape.rank();
        let axis = crate::shape::normalize_axis(axis, rank)?;
        if self.shape()[axis] != 1 {
            return Ok(self.clone());
        }
        let mut new_shape = Vec::with_capacity(rank - 1);
        let mut new_strides = ArrayVec::<[isize; MAX_RANK]>::new();
        for (idx, (&dim, &stride)) in self.shape().iter().zip(self.strides().iter()).enumerate() {
            if idx == axis {
                continue;
            }
            new_shape.push(dim);
            new_strides.push(stride);
        }
        let shape = Shape::from_slice(&new_shape)?;
        Layout::with_strides(shape, new_strides.as_slice(), self.offset_bytes)
    }

    pub fn unsqueeze_axis(&self, axis: isize) -> Result<Self> {
        let rank = self.shape.rank();
        let insert = crate::shape::normalize_axis_inclusive(axis, rank)?;
        if self.is_contiguous() {
            let mut new_dims = self.shape().to_vec();
            new_dims.insert(insert, 1);
            let shape = Shape::from_slice(&new_dims)?;
            return Ok(Layout::contiguous_with_offset(shape, self.offset_bytes));
        }
        let mut new_shape = Vec::with_capacity(rank + 1);
        let mut new_strides = ArrayVec::<[isize; MAX_RANK]>::new();
        for i in 0..insert {
            new_shape.push(self.shape()[i]);
            new_strides.push(self.strides()[i]);
        }
        let inserted_stride = if insert == rank {
            1
        } else {
            self.strides()[insert]
        };
        new_shape.push(1);
        new_strides.push(inserted_stride);
        for i in insert..rank {
            new_shape.push(self.shape()[i]);
            new_strides.push(self.strides()[i]);
        }
        let shape = Shape::from_slice(&new_shape)?;
        Layout::with_strides(shape, new_strides.as_slice(), self.offset_bytes)
    }

    pub fn broadcast_to(&self, new_shape: &Shape) -> Result<Self> {
        let mut new_strides = ArrayVec::<[isize; MAX_RANK]>::new();
        let mut shape_iter = self.shape().iter().rev();
        let mut strides_iter = self.strides().iter().rev();
        for &dim in new_shape.as_slice().iter().rev() {
            let (shape_dim, stride) = match (shape_iter.next(), strides_iter.next()) {
                (Some(&d), Some(&s)) => (d, s),
                (None, None) => (1, 0),
                _ => unreachable!(),
            };
            if shape_dim == dim {
                new_strides.push(stride);
            } else if shape_dim == 1 {
                new_strides.push(0);
            } else {
                return Err(Error::ShapeMismatch {
                    lhs: self.shape().to_vec(),
                    rhs: new_shape.to_vec(),
                });
            }
        }

        if shape_iter.any(|&dim| dim != 1) {
            return Err(Error::ShapeMismatch {
                lhs: self.shape().to_vec(),
                rhs: new_shape.to_vec(),
            });
        }

        new_strides.reverse();
        Self::with_strides(new_shape.clone(), new_strides.as_slice(), self.offset_bytes)
    }

    pub fn broadcast_binary(lhs: &Layout, rhs: &Layout) -> Result<(Layout, Layout)> {
        let new_shape = broadcast_shapes(lhs.shape().as_slice(), rhs.shape().as_slice())?;
        let new_shape = Shape::from_slice(&new_shape)?;
        let lhs = lhs.broadcast_to(&new_shape)?;
        let rhs = rhs.broadcast_to(&new_shape)?;
        Ok((lhs, rhs))
    }

    pub fn iter_offsets(&self, dtype: DType) -> Result<LayoutIter> {
        self.iter_offsets_for(IterMode::Read, dtype)
    }

    pub fn iter_element_indices(&self, dtype: DType) -> Result<impl Iterator<Item = usize> + '_> {
        let elem_size = dtype.size_in_bytes();
        self.iter_offsets(dtype).map(move |iter| {
            iter.map(move |byte_offset| {
                debug_assert_eq!(byte_offset % elem_size, 0);
                byte_offset / elem_size
            })
        })
    }

    pub fn iter_offsets_for(&self, mode: IterMode, dtype: DType) -> Result<LayoutIter> {
        let elem_size = dtype.size_in_bytes() as isize;
        let mut shape = ArrayVec::<[usize; MAX_RANK]>::new();
        let mut indices = ArrayVec::<[usize; MAX_RANK]>::new();
        let mut byte_strides = ArrayVec::<[isize; MAX_RANK]>::new();
        let mut rewinds = ArrayVec::<[isize; MAX_RANK]>::new();

        let base_offset = isize::try_from(self.offset_bytes)
            .map_err(|_| Error::invalid_shape("offset_bytes too large for isize"))?;
        let mut min_offset = base_offset;

        for (i, (&dim, &stride)) in self
            .shape
            .as_slice()
            .iter()
            .zip(self.strides.iter())
            .enumerate()
        {
            shape.push(dim);
            indices.push(0);

            if mode == IterMode::Write && dim > 1 && stride == 0 {
                return Err(Error::invalid_shape(format!(
                    "write mode requires non-zero strides for axes with extent > 1 (axis {} is broadcasted)",
                    i
                )));
            }

            let byte_stride = stride
                .checked_mul(elem_size)
                .ok_or_else(|| Error::invalid_shape("stride overflow"))?;
            byte_strides.push(byte_stride);

            if byte_stride < 0 {
                let max_idx = (dim as isize).saturating_sub(1);
                let drop = byte_stride
                    .checked_mul(max_idx)
                    .ok_or_else(|| Error::invalid_shape("min offset calc overflow"))?;
                min_offset = min_offset
                    .checked_add(drop)
                    .ok_or_else(|| Error::invalid_shape("min offset calc overflow"))?;
            }

            if dim > 0 {
                let rewind = byte_stride
                    .checked_mul((dim - 1) as isize)
                    .ok_or_else(|| Error::invalid_shape("rewind overflow"))?;
                rewinds.push(rewind);
            } else {
                rewinds.push(0);
            }
        }

        if min_offset < 0 {
            return Err(Error::invalid_shape(format!(
                "layout requires accessing negative memory addresses (min_offset={})",
                min_offset
            )));
        }

        Ok(LayoutIter {
            shape,
            indices,
            byte_strides,
            rewinds,
            current_offset: base_offset,
            remaining_elements: self.num_elements(),
        })
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

            let dim_extent = (*dim as isize).saturating_sub(1);
            let extent = stride
                .checked_mul(dim_extent)
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
        shape: Shape,
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
