#![deny(unused_must_use)]

pub mod buffer;
pub mod device;
pub mod dispatcher;
pub mod dtype;
pub mod error;
pub mod layout;
pub mod op;
pub mod shape;
pub mod runtime;
pub mod tensor;

pub use buffer::{BufferId, BufferView};
pub use device::{Device, DeviceKind};
pub use dispatcher::{Dispatcher, KernelLayoutReq};
pub use dtype::{DType, NativeType};
pub use error::{Error, Result};
pub use layout::{Layout, LayoutKind};
pub use op::{OpAttrs, OpKey, OpKind};
pub use runtime::{Runtime, RuntimeBuilder};
pub use tensor::Tensor;
