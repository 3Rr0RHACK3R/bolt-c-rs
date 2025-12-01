use std::{fmt, mem};

use bytemuck::Pod;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    I32,
}

pub const MAX_NATIVE_TYPE_SIZE: usize = mem::size_of::<f64>();

impl DType {
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::F32 => std::mem::size_of::<f32>(),
            DType::F64 => std::mem::size_of::<f64>(),
            DType::I32 => std::mem::size_of::<i32>(),
        }
    }

    pub fn alignment(self) -> usize {
        self.size_in_bytes()
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::I32 => "i32",
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F64)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

pub trait NativeType: Copy + Pod + Send + Sync + 'static + fmt::Debug + Default {
    const DTYPE: DType;
}

pub trait OneValue {
    fn one() -> Self;
}

pub trait ToF32 {
    fn to_f32(self) -> f32;
}

pub trait FloatType: NativeType {}

impl NativeType for f32 {
    const DTYPE: DType = DType::F32;
}

impl FloatType for f32 {}

impl ToF32 for f32 {
    fn to_f32(self) -> f32 {
        self
    }
}

impl OneValue for f32 {
    fn one() -> Self {
        1.0
    }
}

impl NativeType for f64 {
    const DTYPE: DType = DType::F64;
}

impl FloatType for f64 {}

impl ToF32 for f64 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl OneValue for f64 {
    fn one() -> Self {
        1.0
    }
}

impl NativeType for i32 {
    const DTYPE: DType = DType::I32;
}

impl ToF32 for i32 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl OneValue for i32 {
    fn one() -> Self {
        1
    }
}
