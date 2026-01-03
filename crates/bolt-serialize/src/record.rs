use std::borrow::Cow;
use std::path::Path;

use bolt_core::{DType, ParamId, shape::Shape};
use safetensors::tensor::TensorView as SafeTensorView;
use serde::{Deserialize, Serialize};

use crate::load::shard::dtype_from_safe;
use crate::{Error, Result};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    #[serde(rename = "model_param", alias = "param")]
    Param,
    #[serde(rename = "model_buffer", alias = "buffer")]
    Buffer,
    #[serde(rename = "optimizer_state", alias = "optimizer")]
    Optimizer,
    #[serde(rename = "rng_state", alias = "rng")]
    Rng,
    #[default]
    #[serde(rename = "user")]
    User,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordMeta {
    pub name: String,
    pub dtype: DType,
    pub shape: Shape,
    pub role: Role,
    pub group: u32,
    pub param_id: Option<ParamId>,
}

impl RecordMeta {
    pub fn new(name: impl Into<String>, dtype: DType, shape: Shape) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape,
            role: Role::User,
            group: 0,
            param_id: None,
        }
    }

    pub fn with_param_id(mut self, id: ParamId) -> Self {
        self.param_id = Some(id);
        self
    }

    pub fn with_role(mut self, role: Role) -> Self {
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
pub struct Record<'a> {
    pub meta: RecordMeta,
    pub data: Cow<'a, [u8]>,
}

impl<'a> Record<'a> {
    pub fn new(meta: RecordMeta, data: impl Into<Cow<'a, [u8]>>) -> Self {
        Self {
            meta,
            data: data.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RecordView<'a> {
    pub dtype: DType,
    pub shape: Shape,
    pub data: &'a [u8],
}

impl<'a> RecordView<'a> {
    pub(crate) fn from_safetensors_view(
        view: &SafeTensorView<'a>,
        shard_path: &Path,
    ) -> Result<Self> {
        let dtype = dtype_from_safe(view.dtype()).ok_or_else(|| Error::ShardFormat {
            shard: shard_path.to_path_buf(),
            reason: format!("unsupported dtype: {:?}", view.dtype()),
        })?;

        let shape = Shape::from_slice(view.shape()).map_err(|e| Error::ShardFormat {
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
