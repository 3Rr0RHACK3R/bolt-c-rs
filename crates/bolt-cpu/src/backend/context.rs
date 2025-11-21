use std::sync::Arc;

#[derive(Debug)]
pub struct CpuContext {
    // future: pools, tuning knobs, metrics
}

impl CpuContext {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {})
    }
}
