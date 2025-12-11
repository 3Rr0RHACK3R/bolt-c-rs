use crate::Backend;

/// Marker trait for base (non-autodiff) backends.
///
/// Only base backends (Cpu, Cuda) can be wrapped by `Autodiff`.
/// This prevents double-wrapping and ensures parameters are always
/// stored on concrete device backends.
pub trait BaseBackend: Backend {}
