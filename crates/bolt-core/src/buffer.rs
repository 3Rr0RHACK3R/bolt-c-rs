use crate::{dtype::DType, layout::Layout};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferId(u64);

impl BufferId {
    pub fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub fn raw(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug)]
pub struct BufferView {
    pub buffer_id: BufferId,
    pub dtype: DType,
    pub layout: Layout,
}

impl BufferView {
    pub fn num_bytes(&self) -> usize {
        self.layout.num_elements() * self.dtype.size_in_bytes()
    }

    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    pub fn offset_bytes(&self) -> usize {
        self.layout.offset_bytes()
    }
}
