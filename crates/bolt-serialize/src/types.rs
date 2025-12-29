use std::borrow::Cow;
use std::path::Path;

use bolt_core::{DType, shape};
use safetensors::tensor::TensorView as SafeTensorView;
use serde::{Deserialize, Serialize};

use crate::shard::dtype_from_safe;
use crate::{Error, Result};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorRole {
    ModelParam,
    ModelBuffer,
    OptimizerState,
    RngState,
    #[default]
    User,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum ErrorMode {
    #[default]
    Strict,
    Permissive,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: DType,
    pub shape: shape::Shape,
    pub role: TensorRole,
    pub group: u32,
}

impl TensorMeta {
    pub fn new(name: impl Into<String>, dtype: DType, shape: shape::Shape) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape,
            role: TensorRole::User,
            group: 0,
        }
    }

    pub fn with_role(mut self, role: TensorRole) -> Self {
        self.role = role;
        self
    }

    pub fn with_group(mut self, group: u32) -> Self {
        self.group = group;
        self
    }

    pub fn numel(&self) -> Option<u64> {
        self.shape.numel_checked()
    }

    pub fn nbytes(&self) -> Option<u64> {
        self.numel()
            .and_then(|n| n.checked_mul(self.dtype.size_in_bytes() as u64))
    }
}

#[derive(Clone, Debug)]
pub struct TensorToSave<'a> {
    pub meta: TensorMeta,
    pub data: Cow<'a, [u8]>,
}

impl<'a> TensorToSave<'a> {
    pub fn new(meta: TensorMeta, data: impl Into<Cow<'a, [u8]>>) -> Self {
        Self {
            meta,
            data: data.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorView<'a> {
    pub dtype: DType,
    pub shape: shape::Shape,
    pub data: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn from_safetensors_view(view: &SafeTensorView<'a>, shard_path: &Path) -> Result<Self> {
        let dtype = dtype_from_safe(view.dtype()).ok_or_else(|| Error::Safetensors {
            shard: shard_path.to_path_buf(),
            reason: format!("unsupported dtype: {:?}", view.dtype()),
        })?;

        let shape = shape::Shape::from_slice(view.shape()).map_err(|e| Error::Safetensors {
            shard: shard_path.to_path_buf(),
            reason: format!("invalid shape: {e}"),
        })?;

        Ok(Self {
            dtype,
            shape,
            data: view.data(),
        })
    }
}
