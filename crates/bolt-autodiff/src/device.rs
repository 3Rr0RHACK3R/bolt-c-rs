use bolt_core::device::{BackendDevice, DeviceKind};

pub struct AutodiffDevice<Dev> {
    inner: Dev,
}

impl<Dev: BackendDevice> BackendDevice for AutodiffDevice<Dev> {
    fn kind(&self) -> DeviceKind {
        self.inner.kind()
    }
}

impl<Dev: Clone> Clone for AutodiffDevice<Dev> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}
