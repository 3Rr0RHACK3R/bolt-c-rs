use std::sync::Arc;

use bolt_core::{Device, DeviceKind, Result, RuntimeBuilder};

use crate::{CpuDevice, register_cpu_kernels};

pub trait CpuRuntimeBuilderExt {
    fn with_cpu(self) -> Result<RuntimeBuilder>
    where
        Self: Sized;
}

impl CpuRuntimeBuilderExt for RuntimeBuilder {
    fn with_cpu(mut self) -> Result<RuntimeBuilder> {
        register_cpu_kernels(self.dispatcher_mut())?;
        let device = Arc::new(CpuDevice::new()) as Arc<dyn Device>;
        self = self.with_device(DeviceKind::Cpu, device);
        self = self.with_default_device(DeviceKind::Cpu);
        Ok(self)
    }
}
