use std::sync::Arc;

use bolt_core::{
    backend::Backend, device::DeviceKind, dtype::NativeType, error::Result, runtime::RuntimeBuilder,
};

use crate::backend::CpuBackend;

pub trait CpuRuntimeBuilderExt {
    fn with_cpu(self) -> Result<RuntimeBuilder>;
}

impl CpuRuntimeBuilderExt for RuntimeBuilder {
    fn with_cpu(self) -> Result<RuntimeBuilder> {
        register_cpu_backend(self, DeviceKind::Cpu)
    }
}

fn register_cpu_backend(builder: RuntimeBuilder, device: DeviceKind) -> Result<RuntimeBuilder> {
    let backend = Arc::new(CpuBackend::new());
    let builder = register::<CpuBackend, f32>(builder, device, backend.clone())?;
    let builder = register::<CpuBackend, f64>(builder, device, backend.clone())?;
    let builder = register::<CpuBackend, i32>(builder, device, backend)?;
    Ok(builder.with_default_device(device))
}

fn register<B, D>(
    builder: RuntimeBuilder,
    device: DeviceKind,
    backend: Arc<B>,
) -> Result<RuntimeBuilder>
where
    B: Backend<D> + 'static,
    D: NativeType,
{
    builder.register_backend(device, backend)
}
