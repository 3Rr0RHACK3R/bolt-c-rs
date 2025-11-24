use std::fmt;

use bytemuck::cast;

use crate::{
    backend::Backend,
    device::DeviceKind,
    dtype::{DType, NativeType},
    error::Result,
    layout::Layout,
    shape::ConcreteShape,
    tensor::Tensor,
};

const DISPLAY_EDGE_ITEMS: usize = 3;
const DISPLAY_TOTAL_ELEMENT_THRESHOLD: usize = 1000;
const DISPLAY_LINE_WIDTH: usize = 80;

enum DisplayIndex {
    Index(usize),
    Ellipsis,
}

struct TensorFormatter<'a, B, D>
where
    B: Backend<D>,
    D: NativeType,
{
    tensor: &'a Tensor<B, D>,
    scalar_shape: ConcreteShape,
}

impl<'a, B, D> TensorFormatter<'a, B, D>
where
    B: Backend<D>,
    D: NativeType,
{
    fn new(tensor: &'a Tensor<B, D>) -> Self {
        let scalar_shape =
            ConcreteShape::from_slice(&[1]).expect("scalar shape construction must succeed");
        Self {
            tensor,
            scalar_shape,
        }
    }

    fn format(&mut self) -> Result<String> {
        if self.tensor.shape().is_empty() {
            let value = self.read_value(&[])?;
            return Ok(self.format_value(value));
        }

        let truncated = self.tensor.numel() > DISPLAY_TOTAL_ELEMENT_THRESHOLD;
        let mut out = String::new();
        let mut indices = Vec::with_capacity(self.tensor.shape().len());
        self.format_dim(0, &mut indices, truncated, 0, &mut out)?;
        Ok(out)
    }

    fn format_dim(
        &mut self,
        dim: usize,
        indices: &mut Vec<usize>,
        truncated: bool,
        depth: usize,
        out: &mut String,
    ) -> Result<()> {
        out.push('[');
        let len = self.tensor.shape()[dim];
        let entries = self.display_indices(len, truncated);
        let rank = self.tensor.shape().len();
        for (i, entry) in entries.iter().enumerate() {
            if rank == dim + 1 {
                if i > 0 {
                    out.push_str(", ");
                }
                match entry {
                    DisplayIndex::Index(value_idx) => {
                        indices.push(*value_idx);
                        let value = self.read_value(indices)?;
                        out.push_str(&self.format_value(value));
                        indices.pop();
                    }
                    DisplayIndex::Ellipsis => out.push_str("..."),
                }
            } else {
                if i > 0 {
                    out.push(',');
                }
                if depth > 0 || i > 0 {
                    out.push('\n');
                    out.push_str(&indent(depth + 1));
                }
                match entry {
                    DisplayIndex::Index(value_idx) => {
                        indices.push(*value_idx);
                        self.format_dim(dim + 1, indices, truncated, depth + 1, out)?;
                        indices.pop();
                    }
                    DisplayIndex::Ellipsis => out.push_str("..."),
                }
            }
        }
        if rank != dim + 1 {
            out.push('\n');
            out.push_str(&indent(depth));
        }
        out.push(']');
        Ok(())
    }

    fn display_indices(&self, len: usize, truncated: bool) -> Vec<DisplayIndex> {
        if !truncated || len <= DISPLAY_EDGE_ITEMS * 2 {
            return (0..len).map(DisplayIndex::Index).collect();
        }
        let mut out = Vec::with_capacity(DISPLAY_EDGE_ITEMS * 2 + 1);
        for idx in 0..DISPLAY_EDGE_ITEMS {
            out.push(DisplayIndex::Index(idx));
        }
        out.push(DisplayIndex::Ellipsis);
        for idx in (len - DISPLAY_EDGE_ITEMS)..len {
            out.push(DisplayIndex::Index(idx));
        }
        out
    }

    fn read_value(&self, indices: &[usize]) -> Result<D> {
        let offset_bytes = self
            .tensor
            .layout()
            .offset_bytes_for_indices(indices, D::DTYPE)?;
        let layout = Layout::contiguous_with_offset(self.scalar_shape.clone(), offset_bytes);
        let mut value = [D::default(); 1];
        self.tensor
            .backend()
            .read(self.tensor.storage(), &layout, &mut value)?;
        Ok(value[0])
    }

    fn format_value(&self, value: D) -> String {
        match D::DTYPE {
            DType::I32 => {
                let v: i32 = cast(value);
                format!("{v}")
            }
            DType::F32 => {
                let v: f32 = cast(value);
                format_float(v as f64)
            }
            DType::F64 => {
                let v: f64 = cast(value);
                format_float(v)
            }
        }
    }
}

impl<B, D> fmt::Display for Tensor<B, D>
where
    B: Backend<D>,
    D: NativeType + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut formatter = TensorFormatter::new(self);
        let values = formatter.format().map_err(|_| fmt::Error);
        match values {
            Ok(rendered) => write!(
                f,
                "tensor({}, shape={}, dtype={}, device={})",
                rendered,
                format_shape(self.shape()),
                D::DTYPE,
                format_device(self.backend().device_kind())
            ),
            Err(_) => write!(
                f,
                "tensor(<unavailable: device read failed>, shape={}, dtype={}, device={})",
                format_shape(self.shape()),
                D::DTYPE,
                format_device(self.backend().device_kind())
            ),
        }
    }
}

impl<B, D> fmt::Debug for Tensor<B, D>
where
    B: Backend<D>,
    D: NativeType + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

fn format_shape(shape: &[usize]) -> String {
    let mut out = String::from("[");
    for (i, dim) in shape.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&dim.to_string());
    }
    out.push(']');
    out
}

fn format_device(kind: DeviceKind) -> &'static str {
    match kind {
        DeviceKind::Cpu => "cpu",
        DeviceKind::Cuda => "cuda",
    }
}

fn format_float(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-inf".into()
        } else {
            "inf".into()
        };
    }
    let abs = value.abs();
    let use_sci = abs != 0.0 && (abs < 1e-4 || abs >= 1e4);
    let mut repr = if use_sci {
        format!("{value:.6e}")
    } else {
        format!("{value:.6}")
    };
    if let Some(dot) = repr.find('.') {
        let mut end = repr.len();
        while end > dot + 1 && repr.as_bytes()[end - 1] == b'0' {
            end -= 1;
        }
        if end > dot && repr.as_bytes()[end - 1] == b'.' {
            end -= 1;
        }
        repr.truncate(end);
    }
    repr
}

fn indent(depth: usize) -> String {
    const SPACES_PER_LEVEL: usize = 2;
    let width = depth.saturating_mul(SPACES_PER_LEVEL);
    " ".repeat(width.min(DISPLAY_LINE_WIDTH))
}
