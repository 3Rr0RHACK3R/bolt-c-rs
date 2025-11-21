#![deny(unused_must_use)]

pub mod allocator;
pub mod backend;
pub mod device;
pub mod dtype;
pub mod error;
pub mod layout;
pub mod shape;
pub mod storage;
pub mod tensor;

pub use allocator::StorageAllocator;
pub use backend::{Backend, TensorParts};
pub use device::DeviceKind;
pub use dtype::{DType, NativeType, ToF32};
pub use error::{Error, Result};
pub use layout::{Layout, LayoutKind};
pub use storage::{BufferHandle, TensorView};
pub use tensor::Tensor;
