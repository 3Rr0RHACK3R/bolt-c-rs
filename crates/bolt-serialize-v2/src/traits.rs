use crate::{CheckpointReader, CheckpointWriter, Result};

/// Trait for types that can be saved to a checkpoint.
/// This trait is object-safe, allowing use of `&dyn SaveCheckpoint`.
pub trait SaveCheckpoint {
    fn save(&self, w: &mut CheckpointWriter) -> Result<()>;
}

/// Trait for types that can be loaded from a checkpoint.
pub trait LoadCheckpoint {
    fn load(&mut self, r: &mut CheckpointReader) -> Result<()>;
}
