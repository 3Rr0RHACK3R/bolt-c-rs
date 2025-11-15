use crate::{device::DeviceKind, dtype::DType};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum OpKind {
    Fill,
    Copy,
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Exp,
    Relu,
    Sum,
    MatMul,
    Split,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct OpKey {
    pub op: OpKind,
    pub device: DeviceKind,
    pub dtype: DType,
}

#[derive(Clone, Debug)]
pub enum OpAttrs {
    None,
    Reduce(ReduceAttrs),
    Split(SplitAttrs),
}

#[derive(Clone, Debug)]
pub struct ReduceAttrs {
    axes: Vec<usize>,
}

impl OpAttrs {
    pub fn reduce(axes: Vec<usize>) -> Self {
        Self::Reduce(ReduceAttrs { axes })
    }

    pub fn reduce_axes(&self) -> Option<&[usize]> {
        match self {
            OpAttrs::Reduce(attrs) => Some(&attrs.axes),
            _ => None,
        }
    }

    pub fn split(attrs: SplitAttrs) -> Self {
        Self::Split(attrs)
    }

    pub fn split_attrs(&self) -> Option<&SplitAttrs> {
        match self {
            OpAttrs::Split(attrs) => Some(attrs),
            _ => None,
        }
    }
}

impl Default for OpAttrs {
    fn default() -> Self {
        OpAttrs::None
    }
}

#[derive(Clone, Debug)]
pub struct SplitAttrs {
    pub axis: usize,
    pub spec: SplitSpecAttrs,
}

#[derive(Clone, Debug)]
pub enum SplitSpecAttrs {
    ChunkSize { size: usize },
    Sections(Vec<usize>),
}
