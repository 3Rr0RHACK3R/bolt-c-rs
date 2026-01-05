use bolt_core::{BaseBackend, backend::CopyOp, dtype::Float};
use bolt_nn::Store;
use bytemuck;

use crate::{CheckpointWriter, Result, SaveCheckpoint};

impl<B, D> SaveCheckpoint for Store<B, D>
where
    B: BaseBackend + CopyOp<D>,
    D: Float + bytemuck::Pod,
{
    fn save(&self, w: &mut CheckpointWriter) -> Result<()> {
        // Save all parameters
        for (name, param) in self.named_trainable() {
            w.tensor(&name, &param.tensor())?;
        }

        // Save all buffers
        for (name, buffer) in self.named_buffers() {
            w.tensor(&name, &buffer.tensor())?;
        }

        Ok(())
    }
}

impl<B, D> crate::LoadCheckpoint for Store<B, D>
where
    B: BaseBackend + CopyOp<D>,
    D: Float + bytemuck::Pod,
{
    fn load(&mut self, r: &mut crate::CheckpointReader) -> Result<()> {
        let backend = self.backend();

        // Load parameters
        for (name, param) in self.named_trainable() {
            if r.contains(&name) {
                let tensor = r.tensor(&name, &backend)?;
                param.set_tensor(tensor).map_err(|e| {
                    crate::Error::Deserialization(format!(
                        "Failed to set parameter {}: {}",
                        name, e
                    ))
                })?;
            }
        }

        // Load buffers
        for (name, buffer) in self.named_buffers() {
            if r.contains(&name) {
                let tensor = r.tensor(&name, &backend)?;
                buffer.set(tensor).map_err(|e| {
                    crate::Error::Deserialization(format!("Failed to set buffer {}: {}", name, e))
                })?;
            }
        }

        Ok(())
    }
}
