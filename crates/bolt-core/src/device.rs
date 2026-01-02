#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    Cuda,
}

/// Identifier for a device, combining device kind and ordinal.
/// This uniquely identifies a device instance across the system.
/// For CPU, ordinal is always 0. For GPUs, ordinal corresponds to the device index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId {
    kind: DeviceKind,
    ordinal: usize,
}

impl DeviceId {
    /// Create a DeviceId for CPU (ordinal must be 0)
    pub fn cpu() -> Self {
        Self {
            kind: DeviceKind::Cpu,
            ordinal: 0,
        }
    }

    /// Create a DeviceId for a CUDA device
    pub fn cuda(ordinal: usize) -> Self {
        Self {
            kind: DeviceKind::Cuda,
            ordinal,
        }
    }

    /// Returns the device kind
    pub fn kind(&self) -> DeviceKind {
        self.kind
    }

    /// Returns the device ordinal
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }
}

pub trait BackendDevice {
    fn device_id(&self) -> DeviceId;

    fn kind(&self) -> DeviceKind {
        self.device_id().kind()
    }

    fn ordinal(&self) -> usize {
        self.device_id().ordinal()
    }
}
