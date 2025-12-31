use std::collections::BTreeMap;

use bolt_core::backend::CopyOp;
use bolt_core::{BaseBackend, Float};
use bolt_nn::Store;
use bolt_optim::Sgd;
use bolt_tensor::Tensor;

use crate::adapters::optim::OptimizerCheckpointAdapter;
use crate::adapters::tensor::{TensorFromCheckpoint, TensorToRecord};
use crate::{Checkpoint, Record, Result, Role};

fn has_any_velocity_record(ckpt: &Checkpoint) -> bool {
    ckpt.list_by_role(Role::Optimizer)
        .iter()
        .any(|m| m.name.starts_with("optim.") && m.name.ends_with(".vel"))
}

fn format_missing_velocity_keys(keys: &[String], limit: usize) -> String {
    if keys.len() <= limit {
        return keys.join(", ");
    }

    let shown = keys[..limit].join(", ");
    format!("{shown}, … (+{} more)", keys.len() - limit)
}

impl<B, D> OptimizerCheckpointAdapter<B, D> for Sgd<B, D>
where
    B: BaseBackend + CopyOp<D>,
    D: Float + bytemuck::Pod,
{
    fn to_records<'a>(
        &'a self,
        store: &'a Store<B, D>,
    ) -> Box<dyn Iterator<Item = Result<Record<'static>>> + 'a> {
        let group_by_key: BTreeMap<String, u32> = store
            .named_trainable()
            .into_iter()
            .map(|(k, p)| (k, p.group()))
            .collect();

        Box::new(self.velocity_state().iter().map(move |(key, t)| {
            let name = format!("optim.{key}.vel");
            let mut record = t.to_record(&name, Role::Optimizer)?;
            record.meta.group = *group_by_key.get(key).unwrap_or(&0);
            Ok(record)
        }))
    }

    fn restore_from_checkpoint(&mut self, ckpt: &Checkpoint, store: &Store<B, D>) -> Result<()> {
        let backend = store.backend();
        self.velocity_state_mut().clear();

        let has_any_velocity = has_any_velocity_record(ckpt);
        let mut present: Vec<String> = Vec::new();
        let mut missing: Vec<String> = Vec::new();

        for (key, _p) in store.named_trainable() {
            let name = format!("optim.{key}.vel");
            if ckpt.contains(&name) {
                present.push(key);
            } else {
                missing.push(key);
            }
        }

        if has_any_velocity && !missing.is_empty() {
            let msg = format_missing_velocity_keys(&missing, 16);
            return Err(crate::Error::RestoreFailed {
                reason: format!(
                    "checkpoint contains optimizer velocity state, but is missing velocity records for: {msg}"
                ),
            });
        }

        for key in present {
            let name = format!("optim.{key}.vel");
            let t: Tensor<B, D> = Tensor::<B, D>::restore_from_checkpoint(ckpt, &name, &backend)?;
            self.velocity_state_mut().insert(key, t);
        }

        Ok(())
    }
}
