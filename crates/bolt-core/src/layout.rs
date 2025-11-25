use crate::{
    dtype::DType,
    error::{Error, Result},
    shape::{ConcreteShape, MAX_RANK, broadcast_shapes},
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

    pub fn broadcast_to(&self, new_shape: &ConcreteShape) -> Result<Self> {
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
                    rhs: new_shape.as_slice().to_vec(),
                });
            }
        }

        if shape_iter.any(|&dim| dim != 1) {
            return Err(Error::ShapeMismatch {
                lhs: self.shape().to_vec(),
                rhs: new_shape.as_slice().to_vec(),
            });
        }

        new_strides.reverse();
        Self::with_strides(new_shape.clone(), new_strides.as_slice(), self.offset_bytes)
    }

    pub fn broadcast_binary(lhs: &Layout, rhs: &Layout) -> Result<(Layout, Layout)> {
        let new_shape = broadcast_shapes(lhs.shape(), rhs.shape())?;
        let new_shape = ConcreteShape::from_slice(&new_shape)?;
        let lhs = lhs.broadcast_to(&new_shape)?;
        let rhs = rhs.broadcast_to(&new_shape)?;
        Ok((lhs, rhs))
    }

    pub fn iter_offsets(&self, dtype: DType) -> Result<LayoutIter> {
        self.iter_offsets_for(IterMode::Read, dtype)
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
                    "write mode requires non-zero strides for dims > 1 (dim {} is broadcasted)",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_to_scalar() {
        let shape = ConcreteShape::from_slice(&[]).unwrap();
        let layout = Layout::contiguous(shape);
        let new_shape = ConcreteShape::from_slice(&[2, 3]).unwrap();
        let new_layout = layout.broadcast_to(&new_shape).unwrap();
        assert_eq!(new_layout.shape(), &[2, 3]);
        assert_eq!(new_layout.strides(), &[0, 0]);
    }

    #[test]
    fn test_broadcast_to_vector() {
        let shape = ConcreteShape::from_slice(&[3]).unwrap();
        let layout = Layout::contiguous(shape);
        let new_shape = ConcreteShape::from_slice(&[2, 3]).unwrap();
        let new_layout = layout.broadcast_to(&new_shape).unwrap();
        assert_eq!(new_layout.shape(), &[2, 3]);
        assert_eq!(new_layout.strides(), &[0, 1]);
    }

    #[test]
    fn test_broadcast_to_matrix() {
        let shape = ConcreteShape::from_slice(&[2, 1]).unwrap();
        let layout = Layout::contiguous(shape);
        let new_shape = ConcreteShape::from_slice(&[2, 3]).unwrap();
        let new_layout = layout.broadcast_to(&new_shape).unwrap();
        assert_eq!(new_layout.shape(), &[2, 3]);
        assert_eq!(new_layout.strides(), &[1, 0]);
    }

    #[test]
    fn test_broadcast_binary() {
        let shape1 = ConcreteShape::from_slice(&[2, 1]).unwrap();
        let layout1 = Layout::contiguous(shape1);
        let shape2 = ConcreteShape::from_slice(&[3]).unwrap();
        let layout2 = Layout::contiguous(shape2);
        let (new_layout1, new_layout2) = Layout::broadcast_binary(&layout1, &layout2).unwrap();
        assert_eq!(new_layout1.shape(), &[2, 3]);
        assert_eq!(new_layout1.strides(), &[1, 0]);
        assert_eq!(new_layout2.shape(), &[2, 3]);
        assert_eq!(new_layout2.strides(), &[0, 1]);
    }

    #[test]
    fn test_iter_offsets_contiguous() {
        let shape = ConcreteShape::from_slice(&[2, 3]).unwrap();
        let layout = Layout::contiguous(shape);
        let offsets: Vec<usize> = layout.iter_offsets(DType::F32).unwrap().collect();
        assert_eq!(offsets, vec![0, 4, 8, 12, 16, 20]);
    }

    #[test]
    fn test_iter_offsets_broadcasted_read() {
        let shape = ConcreteShape::from_slice(&[2, 1]).unwrap();
        let layout = Layout::contiguous(shape);
        let new_shape = ConcreteShape::from_slice(&[2, 3]).unwrap();
        let new_layout = layout.broadcast_to(&new_shape).unwrap();
        let offsets: Vec<usize> = new_layout.iter_offsets(DType::F32).unwrap().collect();
        assert_eq!(offsets, vec![0, 0, 0, 4, 4, 4]);
    }

    #[test]
    fn test_iter_offsets_broadcasted_write() {
        let shape = ConcreteShape::from_slice(&[2, 1]).unwrap();
        let layout = Layout::contiguous(shape);
        let new_shape = ConcreteShape::from_slice(&[2, 3]).unwrap();
        let new_layout = layout.broadcast_to(&new_shape).unwrap();
        assert!(
            new_layout
                .iter_offsets_for(IterMode::Write, DType::F32)
                .is_err()
        );
    }

    #[test]
    fn test_iter_offsets_transpose() {
        let shape = ConcreteShape::from_slice(&[2, 2]).unwrap();
        let layout = Layout::contiguous(shape);
        // Transpose -> shape [2, 2], strides [4, 8] (since original strides were [8, 4] bytes for F32?)
        // Contiguous F32 2x2: strides [2, 1] elems -> [8, 4] bytes.
        // Transpose: shape [2, 2], strides [1, 2] elems -> [4, 8] bytes.
        let new_layout = layout.transpose(0, 1).unwrap();
        let offsets: Vec<usize> = new_layout.iter_offsets(DType::F32).unwrap().collect();
        // Logical iteration: (0,0), (0,1), (1,0), (1,1)
        // Offset = 0*4 + 0*8 = 0
        // Offset = 0*4 + 1*8 = 8
        // Offset = 1*4 + 0*8 = 4
        // Offset = 1*4 + 1*8 = 12
        assert_eq!(offsets, vec![0, 8, 4, 12]);
    }

    #[test]
    fn test_iter_offsets_negative_stride() {
        // [0, 1, 2, 3, 4] -> slice step -1 -> [4, 3, 2, 1, 0]
        // Currently `Layout::slice` only accepts usize steps, so we manually construct
        // a negative stride layout to verify the iterator handles it correctly.

        let shape = ConcreteShape::from_slice(&[5]).unwrap();
        // Base offset at element 4 -> 4 * 4 = 16 bytes.
        // Stride = -1 elem -> -4 bytes.
        let layout = Layout::with_strides(shape, &[-1], 16).unwrap();
        let offsets: Vec<usize> = layout.iter_offsets(DType::F32).unwrap().collect();
        assert_eq!(offsets, vec![16, 12, 8, 4, 0]);
    }

    #[test]
    fn test_kernel_simulation_usage() {
        // Simulate a kernel that chooses path based on contiguity
        let run_kernel = |layout: &Layout| -> Vec<usize> {
            if layout.is_contiguous() {
                // Fast path: direct slice logic (simulated)
                let start = layout.offset_bytes();
                let end = start + layout.num_elements() * 4; // F32
                (start..end).step_by(4).collect()
            } else {
                // Slow path: iterator
                layout.iter_offsets(DType::F32).unwrap().collect()
            }
        };

        // Case 1: Contiguous
        let shape = ConcreteShape::from_slice(&[2, 2]).unwrap();
        let layout = Layout::contiguous(shape.clone());
        assert_eq!(run_kernel(&layout), vec![0, 4, 8, 12]);

        // Case 2: Strided (Transpose)
        let layout_t = layout.transpose(0, 1).unwrap();
        assert!(!layout_t.is_contiguous());
        assert_eq!(run_kernel(&layout_t), vec![0, 8, 4, 12]);
    }

    #[test]
    fn test_iter_offsets_out_of_bounds_negative() {
        // Construct a layout where negative strides would push the offset below zero.
        // Base offset = 0.
        // Shape = [5], Stride = -1 (elem) -> -4 bytes.
        // 1st element: offset 0
        // 2nd element: offset -4 (Invalid)

        let shape = ConcreteShape::from_slice(&[5]).unwrap();
        // Manually construct with stride -1 and offset 0 (which is too small)
        let layout = Layout::with_strides(shape, &[-1], 0).unwrap();

        let result = layout.iter_offsets(DType::F32);
        assert!(result.is_err());
        match result {
            Err(Error::InvalidShape { message }) => {
                assert!(message.contains("negative memory addresses"));
            }
            _ => panic!("Expected InvalidShape error"),
        }
    }

    #[test]
    fn test_broadcast_to_invalid() {
        let shape = ConcreteShape::from_slice(&[2, 3]).unwrap();
        let layout = Layout::contiguous(shape);
        let new_shape = ConcreteShape::from_slice(&[3]).unwrap();
        assert!(layout.broadcast_to(&new_shape).is_err());
    }
}
