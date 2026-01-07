pub use crate::format::FormatKind;

/// Options for saving a checkpoint.
#[derive(Clone, Debug)]
pub struct CheckpointOptions {
    pub format: FormatKind,
    pub shard_max_bytes: usize,
}

impl Default for CheckpointOptions {
    fn default() -> Self {
        Self {
            format: FormatKind::SafeTensors,
            shard_max_bytes: 512 * 1024 * 1024, // 512 MB default
        }
    }
}

/// Options for loading a checkpoint.
#[derive(Clone, Debug, Default)]
pub struct LoadOpts {
    // Future: could add options like lazy loading preferences, etc.
}
