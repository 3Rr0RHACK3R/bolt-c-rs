use std::collections::BTreeMap;

use bolt_core::backend::CopyOp;
use bolt_core::{BaseBackend, Float, ParamId};
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
        let id_to_name = store.id_to_name();
        let id_to_group: BTreeMap<ParamId, u32> = store
            .named_trainable()
            .into_iter()
            .map(|(_, p)| (p.id(), p.group()))
            .collect();

        Box::new(self.velocity_state().iter().filter_map(move |(id, t)| {
            let param_name = id_to_name.get(id)?;
            let name = format!("optim.{param_name}.vel");
            let mut record = match t.to_record(&name, Role::Optimizer) {
                Ok(r) => r,
                Err(e) => return Some(Err(e)),
            };
            record.meta.group = id_to_group.get(id).copied().unwrap_or(0);
            record.meta.param_id = Some(*id);
            Some(Ok(record))
        }))
    }

    fn restore_from_checkpoint(&mut self, ckpt: &Checkpoint, store: &Store<B, D>) -> Result<()> {
        let backend = store.backend();
        self.velocity_state_mut().clear();

        let _name_to_id = store.name_to_id();
        let has_any_velocity = has_any_velocity_record(ckpt);
        let mut present: Vec<(String, ParamId)> = Vec::new();
        let mut missing: Vec<String> = Vec::new();

        for (key, p) in store.named_trainable() {
            let name = format!("optim.{key}.vel");
            if ckpt.contains(&name) {
                present.push((key, p.id()));
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

        for (key, id) in present {
            let name = format!("optim.{key}.vel");
            let t: Tensor<B, D> = Tensor::<B, D>::restore_from_checkpoint(ckpt, &name, &backend)?;
            self.velocity_state_mut().insert(id, t);
        }

        Ok(())
    }
}
