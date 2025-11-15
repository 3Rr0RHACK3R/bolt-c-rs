use std::sync::Arc;

use bolt_core::Runtime;
use bolt_cpu::CpuRuntimeBuilderExt;

pub fn test_runtime() -> Arc<Runtime> {
    Runtime::builder()
        .with_cpu()
        .expect("register CPU kernels")
        .build()
        .expect("build cpu runtime")
}
