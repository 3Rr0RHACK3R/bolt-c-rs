#![deny(unused_must_use)]

pub mod allocator;
pub mod backend;
mod base_backend;
pub mod device;
pub mod dtype;
pub mod error;
pub mod index;
pub mod layout;
pub mod param_id;
pub mod shape;
pub mod storage;

pub use allocator::{AllocatorDiagnostics, AllocatorSnapshot, DiagnosticsCaps, StorageAllocator};
pub use backend::{Backend, TensorParts};
pub use base_backend::BaseBackend;
pub use device::DeviceKind;
pub use dtype::{CastFrom, DType, Float, NativeType};
pub use error::{Error, Result};
pub use index::TensorIndex;
pub use layout::{Layout, LayoutKind, TensorIndexer};
pub use param_id::{ParamId, ParamIdGen};
pub use storage::TensorView;
