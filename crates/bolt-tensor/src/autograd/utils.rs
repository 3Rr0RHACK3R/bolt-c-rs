use bolt_core::Backend;
use bolt_core::backend::{ReshapeOp, SumOp};
use bolt_core::dtype::NativeType;
use bolt_core::error::{Error, Result};
use bolt_core::shape;

use crate::Tensor;

pub(crate) fn sum_output_shape(
    input: &[usize],
    axes: Option<&[usize]>,
    keepdims: bool,
) -> Vec<usize> {
    match axes {
        None => {
            if keepdims {
                vec![1; input.len()]
            } else {
                vec![]
            }
        }
        Some(axes) if axes.is_empty() => input.to_vec(),
        Some(axes) => {
            if keepdims {
                let mut out = input.to_vec();
                for &a in axes {
                    out[a] = 1;
                }
                out
            } else {
                let mut out = Vec::with_capacity(input.len().saturating_sub(axes.len()));
                for (i, &dim) in input.iter().enumerate() {
                    if axes.binary_search(&i).is_err() {
                        out.push(dim);
                    }
                }
                out
            }
        }
    }
}

pub(crate) fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    let mut acc = 1usize;
    for (i, dim) in shape.iter().enumerate().rev() {
        strides[i] = acc;
        acc = acc.saturating_mul(*dim);
    }
    strides
}

pub(crate) fn reduce_out_flat(
    input_flat: usize,
    input_shape: &[usize],
    input_strides: &[usize],
    output_strides: &[usize],
    axes: Option<&[usize]>,
    keepdims: bool,
) -> usize {
    let in_rank = input_shape.len();

    if axes.is_none() {
        return 0;
    }

    let axes = axes.expect("checked");
    if axes.is_empty() {
        let mut rem = input_flat;
        let mut out_flat = 0usize;
        for i in 0..in_rank {
            let coord = rem / input_strides[i];
            rem %= input_strides[i];
            out_flat += coord * output_strides[i];
        }
        return out_flat;
    }

    let mut rem = input_flat;
    let mut out_flat = 0usize;
    let mut out_dim = 0usize;
    for i in 0..in_rank {
        let coord = rem / input_strides[i];
        rem %= input_strides[i];

        let reduced = axes.binary_search(&i).is_ok();
        if reduced {
            if keepdims {
                out_flat += 0 * output_strides[i];
            }
            continue;
        }

        if keepdims {
            out_flat += coord * output_strides[i];
        } else {
            out_flat += coord * output_strides[out_dim];
            out_dim += 1;
        }
    }
    out_flat
}

pub(crate) fn expand_sum_grad<B, D>(
    backend: &std::sync::Arc<B>,
    grad_output: &[D],
    grad_output_shape: &[usize],
    input_shape: &[usize],
    axes: Option<&[usize]>,
    keepdims: bool,
) -> Result<Tensor<B, D>>
where
    B: Backend,
    D: NativeType,
{
    let in_rank = input_shape.len();
    let out_rank = grad_output_shape.len();

    if axes.is_none() && !keepdims {
        if grad_output.len() != 1 {
            return Err(Error::ShapeMismatch {
                lhs: grad_output_shape.to_vec(),
                rhs: vec![],
            });
        }
        let v = grad_output[0];
        let numel: usize = input_shape.iter().product();
        return Tensor::from_vec(backend, vec![v; numel], input_shape);
    }

    if in_rank == 0 {
        return Tensor::from_vec(backend, grad_output.to_vec(), &[]);
    }

    let in_strides = row_major_strides(input_shape);
    let out_strides = row_major_strides(grad_output_shape);

    let numel: usize = input_shape.iter().product();
    let mut out = vec![D::default(); numel];

    for flat in 0..numel {
        let mut rem = flat;
        let mut out_flat = 0usize;

        let mut out_dim = 0usize;
        for i in 0..in_rank {
            let coord = rem / in_strides[i];
            rem %= in_strides[i];

            let reduced = axes.map(|a| a.binary_search(&i).is_ok()).unwrap_or(true);
            if reduced {
                if keepdims {
                    out_flat += 0 * out_strides[i];
                }
                continue;
            }

            if keepdims {
                out_flat += coord * out_strides[i];
            } else {
                out_flat += coord * out_strides[out_dim];
                out_dim += 1;
            }
        }

        if keepdims && out_rank != in_rank {
            return Err(Error::OpError("sum grad shape mismatch".into()));
        }

        out[flat] = grad_output[out_flat];
    }

    Tensor::from_vec(backend, out, input_shape)
}

pub fn canonical_axes(axes: Option<&[isize]>, rank: usize) -> Result<Option<Vec<usize>>> {
    axes.map(|a| shape::canonical_axes(a, rank)).transpose()
}

pub(crate) fn reduce_grad_to_shape<B, D>(
    grad_output: &Tensor<B, D>,
    input_shape: &[usize],
) -> Result<Tensor<B, D>>
where
    B: Backend + ReshapeOp<D> + SumOp<D> + 'static,
    D: NativeType + 'static,
{
    if grad_output.shape().as_slice() == input_shape {
        return Ok(grad_output.clone());
    }

    let out_shape = grad_output.shape();
    if input_shape.len() > out_shape.len() {
        return Err(Error::ShapeMismatch {
            lhs: input_shape.to_vec(),
            rhs: out_shape.to_vec(),
        });
    }

    let out_rank = out_shape.len();
    let in_rank = input_shape.len();
    let rank_delta = out_rank - in_rank;

    let mut axes: Vec<isize> = Vec::new();
    for out_axis in 0..out_rank {
        let out_dim = out_shape[out_axis];

        let in_dim = if out_axis < rank_delta {
            1
        } else {
            input_shape[out_axis - rank_delta]
        };

        if in_dim == out_dim {
            continue;
        }

        if in_dim == 1 {
            axes.push(out_axis as isize);
            continue;
        }

        return Err(Error::ShapeMismatch {
            lhs: input_shape.to_vec(),
            rhs: out_shape.to_vec(),
        });
    }

    let reduced = if axes.is_empty() {
        grad_output.clone()
    } else {
        let backend = grad_output.backend();
        let parts = backend.sum(
            grad_output.layout(),
            grad_output.storage(),
            Some(&axes),
            true,
        )?;
        Tensor::from_parts(backend, parts.storage, parts.layout)
    };

    if reduced.shape().as_slice() == input_shape {
        Ok(reduced)
    } else {
        reduced.reshape(input_shape)
    }
}
