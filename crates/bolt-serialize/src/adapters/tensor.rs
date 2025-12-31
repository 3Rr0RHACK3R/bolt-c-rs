use std::sync::Arc;

use bolt_core::backend::CopyOp;
use bolt_core::{Backend, NativeType};
use bolt_tensor::Tensor;

use crate::{Checkpoint, Error, Record, RecordMeta, Result, Role};

pub trait TensorToRecord {
    fn to_record(&self, name: &str, role: Role) -> Result<Record<'static>>;
}

pub trait TensorFromCheckpoint<B, D>
where
    B: Backend,
    D: NativeType + bytemuck::Pod,
{
    fn restore_from_checkpoint(
        ckpt: &Checkpoint,
        name: &str,
        backend: &Arc<B>,
    ) -> Result<Tensor<B, D>>;
}

impl<B, D> TensorToRecord for Tensor<B, D>
where
    B: Backend + CopyOp<D>,
    D: NativeType + bytemuck::Pod,
{
    fn to_record(&self, name: &str, role: Role) -> Result<Record<'static>> {
        let data = self.to_vec().map_err(|e| Error::TensorMaterializeFailed {
            name: name.to_string(),
            reason: e.to_string(),
        })?;
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        let meta = RecordMeta::new(name, D::DTYPE, self.shape().clone()).with_role(role);
        Ok(Record::new(meta, bytes))
    }
}

impl<B, D> TensorFromCheckpoint<B, D> for Tensor<B, D>
where
    B: Backend,
    D: NativeType + bytemuck::Pod,
{
    fn restore_from_checkpoint(
        ckpt: &Checkpoint,
        name: &str,
        backend: &Arc<B>,
    ) -> Result<Tensor<B, D>> {
        let view = ckpt.get(name)?;

        if view.dtype != D::DTYPE {
            return Err(Error::DTypeMismatch {
                name: name.to_string(),
                expected: D::DTYPE,
                found: view.dtype,
            });
        }

        let element_size = std::mem::size_of::<D>();
        if view.data.len() % element_size != 0 {
            return Err(Error::ByteSizeNotAligned {
                name: name.to_string(),
                actual: view.data.len() as u64,
                element_size,
                dtype: view.dtype,
            });
        }

        let data: Vec<D> = match bytemuck::try_cast_slice::<u8, D>(view.data) {
            Ok(typed_data) => typed_data.to_vec(),
            Err(bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned) => {
                let mut aligned = vec![0u8; view.data.len()];
                aligned.copy_from_slice(view.data);
                bytemuck::cast_slice::<u8, D>(&aligned).to_vec()
            }
            Err(e) => {
                return Err(Error::TensorRestoreFailed {
                    name: name.to_string(),
                    reason: format!("bytemuck cast failed: {e}"),
                });
            }
        };

        Tensor::from_vec(backend, data, view.shape.as_slice()).map_err(|e| {
            Error::TensorRestoreFailed {
                name: name.to_string(),
                reason: e.to_string(),
            }
        })
    }
}
