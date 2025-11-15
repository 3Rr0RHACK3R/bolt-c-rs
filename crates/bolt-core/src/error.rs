use crate::{device::DeviceKind, dtype::DType, op::OpKind};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid shape: {message}")]
    InvalidShape { message: String },

    #[error("shape mismatch: {lhs:?} vs {rhs:?}")]
    ShapeMismatch { lhs: Vec<usize>, rhs: Vec<usize> },

    #[error("dtype mismatch: {lhs:?} vs {rhs:?}")]
    DTypeMismatch { lhs: DType, rhs: DType },

    #[error("device mismatch: {lhs:?} vs {rhs:?}")]
    DeviceMismatch { lhs: DeviceKind, rhs: DeviceKind },

    #[error("size mismatch: expected {expected}, got {actual}")]
    SizeMismatch { expected: usize, actual: usize },

    #[error("op {op:?} is missing kernel for {device:?}/{dtype:?}")]
    KernelNotFound {
        op: OpKind,
        device: DeviceKind,
        dtype: DType,
    },

    #[error("kernel already registered for {op:?} on {device:?}/{dtype:?}")]
    KernelAlreadyRegistered {
        op: OpKind,
        device: DeviceKind,
        dtype: DType,
    },

    #[error("dtype {dtype:?} is not supported for {op}")]
    UnsupportedDType { op: &'static str, dtype: DType },

    #[error("{op} requires floating point dtype, got {dtype:?}")]
    RequiresFloat { op: &'static str, dtype: DType },

    #[error("invalid axes: {0}")]
    InvalidAxes(String),

    #[error("device error: {0}")]
    Device(String),
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Device(value.to_string())
    }
}

impl Error {
    pub fn invalid_shape(msg: impl Into<String>) -> Self {
        Self::InvalidShape {
            message: msg.into(),
        }
    }
}
