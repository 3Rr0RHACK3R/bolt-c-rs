use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct SaveOpts {
    pub shard_max_bytes: Option<u64>,
    pub alignment: u64,
    pub checksum: bool,
    pub overwrite: bool,
    pub exclude: Vec<String>,
}

impl Default for SaveOpts {
    fn default() -> Self {
        Self {
            shard_max_bytes: Some(2 * 1024 * 1024 * 1024), // 2 GiB
            alignment: 4096,
            checksum: true,
            overwrite: false,
            exclude: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LoadOpts {
    pub lazy: bool,
    pub on_error: OnError,
}

#[derive(Clone, Debug, Default)]
pub enum OnError {
    #[default]
    Fail,
    Skip,
}

#[derive(Clone)]
pub struct RestoreOpts {
    pub strict: bool,
    pub filter: Option<Arc<dyn Fn(&str) -> bool + Send + Sync>>,
    pub rename: Option<Arc<dyn Fn(&str) -> String + Send + Sync>>,
}

impl std::fmt::Debug for RestoreOpts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RestoreOpts")
            .field("strict", &self.strict)
            .field("filter", &self.filter.as_ref().map(|_| "<filter>"))
            .field("rename", &self.rename.as_ref().map(|_| "<rename>"))
            .finish()
    }
}

impl Default for RestoreOpts {
    fn default() -> Self {
        Self {
            strict: true,
            filter: None,
            rename: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RestoreReport {
    pub loaded: Vec<String>,
    pub missing: Vec<String>,
    pub unexpected: Vec<String>,
    pub mismatched: Vec<(String, bolt_core::shape::Shape, bolt_core::shape::Shape)>,
}
