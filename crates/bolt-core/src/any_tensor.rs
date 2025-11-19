use std::any::Any;

use crate::{
    backend::Backend,
    device::{BackendDevice, DeviceKind},
    dtype::{DType, NativeType},
    error::{Error, Result},
    layout::Layout,
    tensor::Tensor,
};

pub struct AnyTensor {
    inner: Box<dyn ErasedTensor>,
}

impl Clone for AnyTensor {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone_box(),
        }
    }
}

impl AnyTensor {
    pub fn new(inner: Box<dyn ErasedTensor>) -> Self {
        Self { inner }
    }

    pub fn from_tensor<B, D>(tensor: Tensor<B, D>) -> Self
    where
        B: Backend<D>,
        D: NativeType,
    {
        Self {
            inner: Box::new(tensor),
        }
    }

    pub fn dtype(&self) -> DType {
        self.inner.dtype()
    }

    pub fn device_kind(&self) -> DeviceKind {
        self.inner.device_kind()
    }

    pub fn layout(&self) -> &Layout {
        self.inner.layout()
    }

    pub fn downcast_ref<B, D>(&self) -> Option<&Tensor<B, D>>
    where
        B: Backend<D>,
        D: NativeType,
    {
        self.inner.as_any().downcast_ref::<Tensor<B, D>>()
    }

    pub fn try_into_tensor<B, D>(self) -> Result<Tensor<B, D>>
    where
        B: Backend<D> + 'static,
        D: NativeType,
    {
        let expected_dtype = D::DTYPE;
        let expected_device = self.device_kind();
        let boxed_any = self.inner.into_any();
        match boxed_any.downcast::<Tensor<B, D>>() {
            Ok(tensor) => Ok(*tensor),
            Err(_) => Err(Error::TensorTypeMismatch {
                dtype: expected_dtype,
                device: expected_device,
            }),
        }
    }
}

pub trait ErasedTensor: Send + Sync {
    fn dtype(&self) -> DType;
    fn device_kind(&self) -> DeviceKind;
    fn layout(&self) -> &Layout;
    fn clone_box(&self) -> Box<dyn ErasedTensor>;
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
    fn as_any(&self) -> &dyn Any;
}

impl<B, D> ErasedTensor for Tensor<B, D>
where
    B: Backend<D>,
    D: NativeType,
{
    fn dtype(&self) -> DType {
        D::DTYPE
    }

    fn device_kind(&self) -> DeviceKind {
        self.device().kind()
    }

    fn layout(&self) -> &Layout {
        self.layout()
    }

    fn clone_box(&self) -> Box<dyn ErasedTensor> {
        Box::new(self.clone())
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl<B, D> From<Tensor<B, D>> for AnyTensor
where
    B: Backend<D>,
    D: NativeType,
{
    fn from(tensor: Tensor<B, D>) -> Self {
        AnyTensor::from_tensor(tensor)
    }
}
