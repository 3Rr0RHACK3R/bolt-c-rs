use crate::{
    device::DeviceKind,
    dtype::DType,
    error::{Error, Result},
};

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

#[derive(Clone, Debug, Default)]
pub enum OpAttrs {
    #[default]
    None,
    Sum {
        axes: Vec<usize>,
    },
    Split {
        axis: usize,
        spec: SplitSpecAttrs,
    },
}

impl OpAttrs {
    pub fn discriminant_name(&self) -> &'static str {
        match self {
            OpAttrs::None => "None",
            OpAttrs::Sum { .. } => "Sum",
            OpAttrs::Split { .. } => "Split",
        }
    }
}

pub trait Operation: Sized + Clone {
    const KIND: OpKind;

    fn to_opattrs(&self) -> OpAttrs;
    fn from_opattrs(attrs: &OpAttrs) -> Result<Self>;
}

#[derive(Clone, Debug)]
pub struct SumOp {
    pub axes: Vec<usize>,
}

impl Operation for SumOp {
    const KIND: OpKind = OpKind::Sum;

    fn to_opattrs(&self) -> OpAttrs {
        OpAttrs::Sum {
            axes: self.axes.clone(),
        }
    }

    fn from_opattrs(attrs: &OpAttrs) -> Result<Self> {
        match attrs {
            OpAttrs::Sum { axes } => Ok(SumOp { axes: axes.clone() }),
            other => Err(Error::OpAttrMismatch {
                op: OpKind::Sum,
                expected: "Sum",
                actual: other.discriminant_name(),
            }),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SplitOp {
    pub axis: usize,
    pub spec: SplitSpecAttrs,
}

impl Operation for SplitOp {
    const KIND: OpKind = OpKind::Split;

    fn to_opattrs(&self) -> OpAttrs {
        OpAttrs::Split {
            axis: self.axis,
            spec: self.spec.clone(),
        }
    }

    fn from_opattrs(attrs: &OpAttrs) -> Result<Self> {
        match attrs {
            OpAttrs::Split { axis, spec } => Ok(SplitOp {
                axis: *axis,
                spec: spec.clone(),
            }),
            other => Err(Error::OpAttrMismatch {
                op: OpKind::Split,
                expected: "Split",
                actual: other.discriminant_name(),
            }),
        }
    }
}

#[derive(Clone, Debug)]
pub enum SplitSpecAttrs {
    ChunkSize { size: usize },
    Sections(Vec<usize>),
}

fn ensure_no_attrs(kind: OpKind, attrs: &OpAttrs) -> Result<()> {
    match attrs {
        OpAttrs::None => Ok(()),
        other => Err(Error::OpAttrMismatch {
            op: kind,
            expected: "None",
            actual: other.discriminant_name(),
        }),
    }
}

macro_rules! define_simple_operation {
    ($name:ident, $kind:expr) => {
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl Operation for $name {
            const KIND: OpKind = $kind;

            fn to_opattrs(&self) -> OpAttrs {
                OpAttrs::None
            }

            fn from_opattrs(attrs: &OpAttrs) -> Result<Self> {
                ensure_no_attrs($kind, attrs)?;
                Ok(Self)
            }
        }
    };
}

define_simple_operation!(FillOp, OpKind::Fill);
define_simple_operation!(CopyOp, OpKind::Copy);
define_simple_operation!(AddOp, OpKind::Add);
define_simple_operation!(SubOp, OpKind::Sub);
define_simple_operation!(MulOp, OpKind::Mul);
define_simple_operation!(DivOp, OpKind::Div);
define_simple_operation!(NegOp, OpKind::Neg);
define_simple_operation!(ExpOp, OpKind::Exp);
define_simple_operation!(ReluOp, OpKind::Relu);
define_simple_operation!(MatMulOp, OpKind::MatMul);
