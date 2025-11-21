use crate::{
    device::DeviceKind,
    dtype::DType,
    error::{Error, Result},
    layout::Layout,
};

/// Lightweight handle that carries buffer metadata needed for safety checks.
#[derive(Clone, Debug)]
pub struct BufferHandle {
    device: DeviceKind,
    dtype: DType,
    len_bytes: usize,
}

impl BufferHandle {
    pub fn new(device: DeviceKind, dtype: DType, len_bytes: usize) -> Result<Self> {
        if len_bytes == 0 {
            return Err(Error::invalid_shape("buffer size must be > 0"));
        }
        Ok(Self {
            device,
            dtype,
            len_bytes,
        })
    }

    pub fn device_kind(&self) -> DeviceKind {
        self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn len_bytes(&self) -> usize {
        self.len_bytes
    }

    pub fn validate_layout(&self, layout: &Layout) -> Result<()> {
        layout.validate_bounds(self.dtype, self.len_bytes)
    }
}

/// Borrowed view passed into kernels (non-owning).
#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a, S> {
    pub storage: &'a S,
    pub layout: &'a Layout,
}

impl<'a, S> TensorView<'a, S> {
    pub fn new(storage: &'a S, layout: &'a Layout) -> Self {
        Self { storage, layout }
    }
}
