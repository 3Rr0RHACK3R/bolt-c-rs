use std::sync::{Arc, Once};

use bolt_core::{device::Device, dispatcher::init_dispatcher};
use bolt_cpu::{CpuDevice, register_cpu_kernels};

static DISPATCH_ONCE: Once = Once::new();

pub fn init_cpu_dispatcher() {
    DISPATCH_ONCE.call_once(|| {
        init_dispatcher(|dispatcher| register_cpu_kernels(dispatcher))
            .expect("dispatcher already initialized");
    });
}

pub fn test_device() -> Arc<dyn Device> {
    Arc::new(CpuDevice::new()) as Arc<dyn Device>
}
