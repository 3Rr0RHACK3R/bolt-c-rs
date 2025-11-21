#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    Cuda,
}

pub trait BackendDevice {
    fn kind(&self) -> DeviceKind;
}
