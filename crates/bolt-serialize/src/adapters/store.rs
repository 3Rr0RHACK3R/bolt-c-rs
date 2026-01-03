use std::collections::{BTreeSet, HashMap};

use bolt_core::backend::CopyOp;
use bolt_core::{BaseBackend, Float};
use bolt_nn::{Buffer, Param, Store};
use bolt_tensor::Tensor;

use crate::adapters::tensor::{TensorFromCheckpoint, TensorToRecord};
use crate::{Checkpoint, Record, RestoreOpts, RestoreReport, Result, Role};

pub trait StoreCheckpointAdapter {
    fn to_records(&self) -> Box<dyn Iterator<Item = Result<Record<'static>>> + '_>;

    fn restore_from_checkpoint(
        &self,
        ckpt: &Checkpoint,
        opts: &RestoreOpts,
    ) -> Result<RestoreReport>;
}

impl<B, D> StoreCheckpointAdapter for Store<B, D>
where
    B: BaseBackend + CopyOp<D>,
    D: Float + bytemuck::Pod,
{
    fn to_records(&self) -> Box<dyn Iterator<Item = Result<Record<'static>>> + '_> {
        let params = self.named_trainable().into_iter().map(|(name, p)| {
            let mut record = p.tensor().to_record(&name, Role::Param)?;
            record.meta.group = p.group();
            record.meta.param_id = Some(p.id());
            Ok(record)
        });

        let buffers = self.named_buffers().into_iter().map(|(name, b)| {
            let mut record = b.tensor().to_record(&name, Role::Buffer)?;
            record.meta.group = b.group();
            record.meta.param_id = Some(b.id());
            Ok(record)
        });

        Box::new(params.chain(buffers))
    }

    fn restore_from_checkpoint(
        &self,
        ckpt: &Checkpoint,
        opts: &RestoreOpts,
    ) -> Result<RestoreReport> {
        let mut used = BTreeSet::new();
        let mut report = RestoreReport::default();

        let params_by_key: HashMap<String, Param<B, D>> =
            self.named_trainable().into_iter().collect();
        let buffers_by_key: HashMap<String, Buffer<B, D>> =
            self.named_buffers().into_iter().collect();

        let backend = self.backend();

        for meta in ckpt.list() {
            if !matches!(meta.role, Role::Param | Role::Buffer) {
                continue;
            }

            if let Some(filter) = &opts.filter
                && !(filter)(&meta.name)
            {
                continue;
            }

            let target_key = match &opts.rename {
                None => meta.name.clone(),
                Some(rename) => (rename)(&meta.name),
            };

            let target_kind = if params_by_key.contains_key(&target_key) {
                Role::Param
            } else if buffers_by_key.contains_key(&target_key) {
                Role::Buffer
            } else {
                report.unexpected.push(meta.name.clone());
                continue;
            };

            used.insert(target_key.clone());

            let expected_shape = match target_kind {
                Role::Param => params_by_key[&target_key].shape().clone(),
                Role::Buffer => buffers_by_key[&target_key].shape().clone(),
                _ => unreachable!("checked above"),
            };

            if expected_shape != meta.shape {
                report
                    .mismatched
                    .push((target_key.clone(), expected_shape, meta.shape.clone()));
                continue;
            }

            let tensor: Tensor<B, D> =
                Tensor::<B, D>::restore_from_checkpoint(ckpt, &meta.name, &backend)?;

            match target_kind {
                Role::Param => {
                    let p = &params_by_key[&target_key];
                    p.set_group(meta.group);
                    p.set_tensor(tensor)
                        .map_err(|e| crate::Error::RestoreFailed {
                            reason: format!("failed to restore param '{target_key}': {e}"),
                        })?;
                }
                Role::Buffer => {
                    let b = &buffers_by_key[&target_key];
                    b.set_group(meta.group);
                    b.set(tensor).map_err(|e| crate::Error::RestoreFailed {
                        reason: format!("failed to restore buffer '{target_key}': {e}"),
                    })?;
                }
                _ => unreachable!("checked above"),
            }

            report.loaded.push(target_key);
        }

        for k in self.expected_keys() {
            if !used.contains(&k) {
                report.missing.push(k);
            }
        }

        if opts.strict
            && (!report.missing.is_empty()
                || !report.unexpected.is_empty()
                || !report.mismatched.is_empty())
        {
            return Err(crate::Error::RestoreFailed {
                reason: format!("strict restore failed: {report:?}"),
            });
        }

        Ok(report)
    }
}
