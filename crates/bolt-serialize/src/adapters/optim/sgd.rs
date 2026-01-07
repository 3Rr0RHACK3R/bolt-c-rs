use bolt_core::{BaseBackend, backend::CopyOp, dtype::Float};
use bolt_optim::Sgd;
use bytemuck;

use crate::{CheckpointWriter, Result, SaveCheckpoint};

impl<B, D> SaveCheckpoint for Sgd<B, D>
where
    B: BaseBackend + CopyOp<D>,
    D: Float + bytemuck::Pod,
{
    fn save(&self, w: &mut CheckpointWriter) -> Result<()> {
        // Save velocity state by parameter name (stable across restarts)
        for (name, vel) in self.velocity_state() {
            let key = format!("vel.{}", name);
            w.tensor(&key, vel)?;
        }
        Ok(())
    }
}

impl<B, D> crate::LoadCheckpoint for Sgd<B, D>
where
    B: BaseBackend + CopyOp<D>,
    D: Float + bytemuck::Pod,
{
    fn load(&mut self, r: &mut crate::CheckpointReader) -> Result<()> {
        let backend = self.backend();

        // Load velocity state by matching parameter names
        // Find all velocity keys and match them to optimizer's parameter names
        let vel_keys: Vec<String> = r
            .keys()
            .into_iter()
            .filter(|k| k.starts_with("vel."))
            .collect();

        let mut loaded = std::collections::HashSet::new();

        for key in vel_keys {
            // Extract parameter name from key (format: "vel.{name}")
            let name = key.strip_prefix("vel.").ok_or_else(|| {
                crate::Error::Deserialization(format!("Invalid velocity key format: {}", key))
            })?;

            // Load tensor for this parameter name
            let tensor = r.tensor(&key, &backend)?;
            self.velocity_state_mut().insert(name.to_string(), tensor);
            loaded.insert(name.to_string());
        }

        // Clear any velocity states that weren't loaded (they'll be recreated on first step)
        self.velocity_state_mut()
            .retain(|name, _| loaded.contains(name));

        Ok(())
    }
}
