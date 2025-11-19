#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    Cuda,
}

// Legacy Device trait removed.
