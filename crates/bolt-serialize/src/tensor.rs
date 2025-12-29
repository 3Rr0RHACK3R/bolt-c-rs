use std::path::Path;
use std::sync::Arc;

use bolt_core::backend::CopyOp;
use bolt_core::{Backend, NativeType};
use bolt_tensor::Tensor;

use crate::{
    Error, Result, TensorMeta, TensorRole, TensorSet, TensorSetLoadOptions, TensorSetSaveOptions,
    TensorToSave, load_tensor_set, save_tensor_set,
};

fn prepare_tensor_to_save<B, D>(
    name: &str,
    tensor: &Tensor<B, D>,
    role: Option<TensorRole>,
    out_dir: &Path,
) -> Result<TensorToSave<'static>>
where
    B: Backend + CopyOp<D>,
    D: NativeType,
{
    let data = tensor.to_vec().map_err(|e| Error::Safetensors {
        shard: out_dir.to_path_buf(),
        reason: e.to_string(),
    })?;
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let mut meta = TensorMeta::new(name, D::DTYPE, tensor.shape().clone());
    if let Some(r) = role {
        meta = meta.with_role(r);
    }
    Ok(TensorToSave::new(meta, bytes))
}

pub fn save<B, D>(
    name: &str,
    tensor: &Tensor<B, D>,
    out_dir: &Path,
    opts: &TensorSetSaveOptions,
) -> Result<()>
where
    B: Backend + CopyOp<D>,
    D: NativeType,
{
    let tensor_to_save = prepare_tensor_to_save(name, tensor, None, out_dir)?;
    save_tensor_set([tensor_to_save], out_dir, opts)
}

pub fn save_many<'a, B, D, I>(tensors: I, out_dir: &Path, opts: &TensorSetSaveOptions) -> Result<()>
where
    B: Backend + CopyOp<D>,
    D: NativeType,
    I: IntoIterator<Item = (&'a str, &'a Tensor<B, D>)>,
{
    let tensors_to_save: Vec<TensorToSave<'static>> = tensors
        .into_iter()
        .map(|(name, tensor)| prepare_tensor_to_save(name, tensor, None, out_dir))
        .collect::<Result<Vec<_>>>()?;

    save_tensor_set(tensors_to_save, out_dir, opts)
}

pub fn save_with_role<B, D>(
    name: &str,
    tensor: &Tensor<B, D>,
    role: TensorRole,
    out_dir: &Path,
    opts: &TensorSetSaveOptions,
) -> Result<()>
where
    B: Backend + CopyOp<D>,
    D: NativeType,
{
    let tensor_to_save = prepare_tensor_to_save(name, tensor, Some(role), out_dir)?;
    save_tensor_set([tensor_to_save], out_dir, opts)
}

pub fn load<B, D>(
    name: &str,
    dir: &Path,
    backend: &Arc<B>,
    opts: &TensorSetLoadOptions,
) -> Result<Tensor<B, D>>
where
    B: Backend,
    D: NativeType,
{
    let set = load_tensor_set(dir, opts)?;
    load_from_set(name, &set, backend)
}

pub fn load_from_set<B, D>(name: &str, set: &TensorSet, backend: &Arc<B>) -> Result<Tensor<B, D>>
where
    B: Backend,
    D: NativeType,
{
    let view = set.get(name)?;

    if view.dtype != D::DTYPE {
        return Err(Error::DTypeMismatch {
            name: name.to_string(),
            expected: D::DTYPE,
            found: view.dtype,
        });
    }

    let element_size = std::mem::size_of::<D>();
    if view.data.len() % element_size != 0 {
        return Err(Error::ByteSizeMismatch {
            name: name.to_string(),
            expected: (view.numel() * element_size as u64),
            actual: view.data.len() as u64,
            numel: view.numel(),
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
            return Err(Error::Safetensors {
                shard: set.artifact_dir().to_path_buf(),
                reason: format!("bytemuck cast failed: {e}"),
            });
        }
    };

    Tensor::from_vec(backend, data, view.shape.as_slice()).map_err(|e| Error::Safetensors {
        shard: set.artifact_dir().to_path_buf(),
        reason: e.to_string(),
    })
}

pub fn load_set(dir: &Path, opts: &TensorSetLoadOptions) -> Result<TensorSet> {
    load_tensor_set(dir, opts)
}
