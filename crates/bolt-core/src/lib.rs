#![deny(unused_must_use)]

pub mod allocator;
pub mod any_tensor;
pub mod backend;
pub mod device;
pub mod dtype;
pub mod error;
pub mod layout;
pub mod runtime;
pub mod shape;
pub mod tensor;

pub use allocator::{AllocatorHandle, AllocatorMetrics, StorageAllocator, StorageBlock};
pub use any_tensor::AnyTensor;
pub use backend::Backend;
pub use device::DeviceKind;
pub use dtype::{DType, NativeType};
pub use error::{Error, Result};
pub use layout::{Layout, LayoutKind};
pub use runtime::{BackendRegistry, Runtime, RuntimeBuilder};
pub use tensor::Tensor;
