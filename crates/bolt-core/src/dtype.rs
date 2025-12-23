use std::{fmt, mem};

use bytemuck::Pod;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    U8,
    I32,
    I64,
    F32,
    F64,
}

pub const MAX_NATIVE_TYPE_SIZE: usize = mem::size_of::<f64>();

impl DType {
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::U8 => std::mem::size_of::<u8>(),
            DType::I32 => std::mem::size_of::<i32>(),
            DType::I64 => std::mem::size_of::<i64>(),
            DType::F32 => std::mem::size_of::<f32>(),
            DType::F64 => std::mem::size_of::<f64>(),
        }
    }

    pub fn alignment(self) -> usize {
        self.size_in_bytes()
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::U8 => "u8",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::F32 => "f32",
            DType::F64 => "f64",
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
    const SUPPORTS_GRAD: bool = false;

    fn one() -> Self;
    fn from_usize(n: usize) -> Self;
}

pub trait CastFrom<Src: NativeType>: NativeType {
    fn cast_from(src: Src) -> Self;
}

macro_rules! impl_cast_from {
    ($dst:ty; $($src:ty),* $(,)?) => {
        $(
            impl CastFrom<$src> for $dst {
                fn cast_from(src: $src) -> Self {
                    src as $dst
                }
            }
        )*
    };
}

impl_cast_from!(u8; u8, i32, i64, f32, f64);
impl_cast_from!(i32; u8, i32, i64, f32, f64);
impl_cast_from!(i64; u8, i32, i64, f32, f64);
impl_cast_from!(f32; u8, i32, i64, f32, f64);
impl_cast_from!(f64; u8, i32, i64, f32, f64);

pub trait Float:
    NativeType
    + std::ops::Neg<Output = Self>
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
{
    fn zero() -> Self;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;

    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn tanh(self) -> Self;
    fn powf(self, exp: Self) -> Self;
    fn abs(self) -> Self;
}

impl NativeType for f32 {
    const DTYPE: DType = DType::F32;
    const SUPPORTS_GRAD: bool = true;

    fn one() -> Self {
        1.0
    }

    fn from_usize(n: usize) -> Self {
        n as f32
    }
}

impl Float for f32 {
    fn zero() -> Self {
        0.0
    }

    fn from_f64(v: f64) -> Self {
        v as f32
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn sin(self) -> Self {
        f32::sin(self)
    }

    fn cos(self) -> Self {
        f32::cos(self)
    }

    fn exp(self) -> Self {
        f32::exp(self)
    }

    fn ln(self) -> Self {
        f32::ln(self)
    }

    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }

    fn tanh(self) -> Self {
        f32::tanh(self)
    }

    fn powf(self, exp: Self) -> Self {
        f32::powf(self, exp)
    }

    fn abs(self) -> Self {
        f32::abs(self)
    }
}

impl NativeType for f64 {
    const DTYPE: DType = DType::F64;
    const SUPPORTS_GRAD: bool = true;

    fn one() -> Self {
        1.0
    }

    fn from_usize(n: usize) -> Self {
        n as f64
    }
}

impl Float for f64 {
    fn zero() -> Self {
        0.0
    }

    fn from_f64(v: f64) -> Self {
        v
    }

    fn to_f64(self) -> f64 {
        self
    }

    fn sin(self) -> Self {
        f64::sin(self)
    }

    fn cos(self) -> Self {
        f64::cos(self)
    }

    fn exp(self) -> Self {
        f64::exp(self)
    }

    fn ln(self) -> Self {
        f64::ln(self)
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    fn tanh(self) -> Self {
        f64::tanh(self)
    }

    fn powf(self, exp: Self) -> Self {
        f64::powf(self, exp)
    }

    fn abs(self) -> Self {
        f64::abs(self)
    }
}

impl NativeType for i32 {
    const DTYPE: DType = DType::I32;

    fn one() -> Self {
        1
    }

    fn from_usize(n: usize) -> Self {
        n as i32
    }
}

impl NativeType for i64 {
    const DTYPE: DType = DType::I64;

    fn one() -> Self {
        1
    }

    fn from_usize(n: usize) -> Self {
        n as i64
    }
}

impl NativeType for u8 {
    const DTYPE: DType = DType::U8;

    fn one() -> Self {
        1
    }

    fn from_usize(n: usize) -> Self {
        n as u8
    }
}
