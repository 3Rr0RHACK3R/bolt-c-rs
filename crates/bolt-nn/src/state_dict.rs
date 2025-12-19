use std::collections::{BTreeMap, BTreeSet};
use std::mem;
use std::sync::Arc;

use bolt_autodiff::Float;
use bolt_core::backend::CopyOp;
use bolt_core::dtype::DType;
use bolt_core::{BaseBackend, Tensor};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::visit::{FlatNamed, FlatNamedMut, NamedParams};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorData {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

pub type StateDict = BTreeMap<String, TensorData>;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadReport {
    pub missing_keys: Vec<String>,
    pub unexpected_keys: Vec<String>,
    pub shape_mismatches: Vec<ShapeMismatch>,
    pub dtype_mismatches: Vec<DTypeMismatch>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeMismatch {
    pub key: String,
    pub expected: Vec<usize>,
    pub got: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DTypeMismatch {
    pub key: String,
    pub expected: DType,
    pub got: DType,
}

pub fn state_dict<B, D, T>(module: &T) -> Result<StateDict>
where
    B: BaseBackend + CopyOp<D>,
    D: Float,
    T: NamedParams<B, D>,
{
    let mut out = StateDict::new();

    let mut err: Option<Error> = None;
    module.visit_named_params(&mut |key, p| {
        if err.is_some() {
            return;
        }
        match tensor_data_from_tensor::<B, D>(p.tensor()) {
            Ok(data) => {
                if out.insert(key.to_string(), data).is_some() {
                    err = Some(Error::MissingParam(format!(
                        "duplicate state_dict key '{key}'"
                    )));
                }
            }
            Err(e) => {
                err = Some(e);
            }
        }
    });

    if let Some(e) = err {
        return Err(e);
    }

    Ok(out)
}

pub fn state_dict_flat<B, D, T>(module: &T) -> Result<StateDict>
where
    B: BaseBackend + CopyOp<D>,
    D: Float,
    T: bolt_autodiff::HasParams<B, D>,
{
    state_dict::<B, D, FlatNamed<'_, B, D, T>>(&FlatNamed::new(module))
}

pub fn load_state_dict<B, D, T>(
    module: &mut T,
    backend: &Arc<B>,
    dict: &StateDict,
    strict: bool,
) -> Result<LoadReport>
where
    B: BaseBackend + CopyOp<D>,
    D: Float,
    T: NamedParams<B, D>,
{
    let mut report = LoadReport::default();
    let mut visited = BTreeSet::new();

    let mut err: Option<Error> = None;
    module.visit_named_params_mut(&mut |key, p| {
        visited.insert(key.to_string());
        let Some(entry) = dict.get(key) else {
            report.missing_keys.push(key.to_string());
            return;
        };

        if entry.dtype != D::DTYPE {
            report.dtype_mismatches.push(DTypeMismatch {
                key: key.to_string(),
                expected: D::DTYPE,
                got: entry.dtype,
            });
            return;
        }

        let expected_shape = p.tensor().shape().to_vec();
        if entry.shape != expected_shape {
            report.shape_mismatches.push(ShapeMismatch {
                key: key.to_string(),
                expected: expected_shape,
                got: entry.shape.clone(),
            });
            return;
        }

        match tensor_from_tensor_data::<B, D>(backend, entry) {
            Ok(t) => {
                *p.tensor_mut() = t;
            }
            Err(e) => {
                if strict {
                    err = Some(e);
                } else {
                    report.missing_keys.push(key.to_string());
                }
            }
        }
    });

    for key in dict.keys() {
        if !visited.contains(key) {
            report.unexpected_keys.push(key.clone());
        }
    }

    if strict {
        if let Some(e) = err {
            return Err(e);
        }
        if let Some(m) = report.dtype_mismatches.first() {
            return Err(Error::MissingParam(format!(
                "dtype mismatch for '{}': expected {}, got {}",
                m.key, m.expected, m.got
            )));
        }
        if let Some(m) = report.shape_mismatches.first() {
            return Err(Error::ShapeMismatch {
                expected: m.expected.clone(),
                got: m.got.clone(),
            });
        }
        if let Some(key) = report.missing_keys.first() {
            return Err(Error::MissingParam(format!("missing key '{key}'")));
        }
        if let Some(key) = report.unexpected_keys.first() {
            return Err(Error::MissingParam(format!("unexpected key '{key}'")));
        }
    }

    Ok(report)
}

pub fn load_state_dict_flat<B, D, T>(
    module: &mut T,
    backend: &Arc<B>,
    dict: &StateDict,
    strict: bool,
) -> Result<LoadReport>
where
    B: BaseBackend + CopyOp<D>,
    D: Float,
    T: bolt_autodiff::HasParams<B, D>,
{
    load_state_dict::<B, D, FlatNamedMut<'_, B, D, T>>(
        &mut FlatNamedMut::new(module),
        backend,
        dict,
        strict,
    )
}

fn tensor_data_from_tensor<B, D>(tensor: &Tensor<B, D>) -> Result<TensorData>
where
    B: BaseBackend + CopyOp<D>,
    D: Float,
{
    let values = tensor.to_vec()?;
    let bytes = slice_as_bytes(&values);
    Ok(TensorData {
        dtype: D::DTYPE,
        shape: tensor.shape().to_vec(),
        bytes,
    })
}

fn tensor_from_tensor_data<B, D>(backend: &Arc<B>, data: &TensorData) -> Result<Tensor<B, D>>
where
    B: BaseBackend,
    D: Float,
{
    if data.dtype != D::DTYPE {
        return Err(Error::MissingParam(format!(
            "dtype mismatch: expected {}, got {}",
            D::DTYPE,
            data.dtype
        )));
    }

    let values = bytes_as_vec::<D>(&data.bytes)?;
    let expected_numel: usize = data.shape.iter().product();
    if values.len() != expected_numel {
        return Err(Error::MissingParam(format!(
            "byte payload size mismatch: expected {expected_numel} elements, got {}",
            values.len()
        )));
    }

    Tensor::from_vec(backend, values, &data.shape).map_err(Into::into)
}

fn slice_as_bytes<T: Copy>(slice: &[T]) -> Vec<u8> {
    let len_bytes = slice.len() * mem::size_of::<T>();
    let ptr = slice.as_ptr().cast::<u8>();
    // SAFETY: T is `Copy`, and we only read its raw representation.
    unsafe { std::slice::from_raw_parts(ptr, len_bytes) }.to_vec()
}

fn bytes_as_vec<T: Copy>(bytes: &[u8]) -> Result<Vec<T>> {
    let elem = mem::size_of::<T>();
    if elem == 0 {
        return Err(Error::MissingParam("zero-sized dtype unsupported".into()));
    }
    if bytes.len() % elem != 0 {
        return Err(Error::MissingParam("invalid byte length for dtype".into()));
    }
    let len = bytes.len() / elem;

    let mut out: Vec<T> = Vec::with_capacity(len);
    for i in 0..len {
        let start = i * elem;
        let src = &bytes[start..start + elem];
        let mut tmp = mem::MaybeUninit::<T>::uninit();
        // SAFETY: tmp is properly aligned for T; we copy exact byte size.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), tmp.as_mut_ptr().cast::<u8>(), elem);
            out.push(tmp.assume_init());
        }
    }

    Ok(out)
}
