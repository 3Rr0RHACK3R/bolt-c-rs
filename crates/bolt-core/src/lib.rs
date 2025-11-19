#![deny(unused_must_use)]

pub mod allocator;
pub mod backend;
pub mod device;
pub mod dtype;
pub mod error;
pub mod layout;
pub mod shape;
pub mod tensor;

pub use allocator::{AllocatorHandle, AllocatorMetrics, StorageAllocator, StorageBlock};
pub use backend::Backend;
pub use device::DeviceKind;
pub use dtype::{DType, NativeType};
pub use error::{Error, Result};
pub use layout::{Layout, LayoutKind};
pub use tensor::Tensor;
