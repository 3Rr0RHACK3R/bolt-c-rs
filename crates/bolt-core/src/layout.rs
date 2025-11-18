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

    pub fn slice(
        &self,
        axis: usize,
        start: usize,
        end: usize,
        step: usize,
        dtype: DType,
    ) -> Result<Self> {
        if axis >= self.shape.rank() {
            return Err(Error::invalid_shape(format!(
                "axis {axis} out of bounds for rank {}",
                self.shape.rank()
            )));
        }
        if step == 0 {
            return Err(Error::invalid_shape("slice step must be > 0"));
        }
        let dim = self.shape()[axis];
        if start >= end || end > dim {
            return Err(Error::invalid_shape(format!(
                "invalid slice range [{start}, {end}) for dim {dim}"
            )));
        }
        let new_len = (end - start).div_ceil(step);
        let mut new_shape = self.shape().to_vec();
        new_shape[axis] = new_len;
        let mut new_strides = self.strides;
        new_strides[axis] *= step as isize;
        let stride = self.strides()[axis];
        let offset_bytes =
            Self::validate_slice_offset_bytes(stride, start, dtype, self.offset_bytes())?;
        let shape = ConcreteShape::from_slice(&new_shape)?;
        Layout::with_strides(shape, new_strides.as_slice(), offset_bytes)
    }

    fn validate_slice_offset_bytes(
        stride: isize,
        start: usize,
        dtype: DType,
        base_offset: usize,
    ) -> Result<usize> {
        let start_isize = isize::try_from(start)
            .map_err(|_| Error::invalid_shape("slice start exceeds addressable range"))?;
        let elem_offset = stride
            .checked_mul(start_isize)
            .ok_or_else(|| Error::invalid_shape("slice offset overflow (stride*start)"))?;
        let byte_delta = elem_offset
            .checked_mul(dtype.size_in_bytes() as isize)
            .ok_or_else(|| Error::invalid_shape("slice offset overflow (bytes)"))?;
        base_offset
            .checked_add_signed(byte_delta)
            .ok_or_else(|| Error::invalid_shape("slice offset overflow (base+delta)"))
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
