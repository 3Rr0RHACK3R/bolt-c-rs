use std::{any::Any, collections::HashMap, marker::PhantomData, sync::Arc};

use bytemuck::try_cast_slice;

use crate::{
    any_tensor::AnyTensor,
    backend::Backend,
    device::DeviceKind,
    dtype::{DType, NativeType},
    error::{Error, Result},
    tensor::Tensor,
};

trait ErasedBackendSlot: Send + Sync {
    fn backend_any(&self) -> Arc<dyn Any + Send + Sync>;
    fn from_slice(&self, shape: &[usize], bytes: &[u8]) -> Result<AnyTensor>;
    fn zeros(&self, shape: &[usize]) -> Result<AnyTensor>;
}

struct TypedBackendSlot<B, D>
where
    B: Backend<D> + 'static,
    D: NativeType,
{
    device: DeviceKind,
    backend: Arc<B>,
    _marker: PhantomData<D>,
}

impl<B, D> TypedBackendSlot<B, D>
where
    B: Backend<D> + 'static,
    D: NativeType,
{
    fn new(device: DeviceKind, backend: Arc<B>) -> Self {
        Self {
            device,
            backend,
            _marker: PhantomData,
        }
    }
}

impl<B, D> ErasedBackendSlot for TypedBackendSlot<B, D>
where
    B: Backend<D> + 'static,
    D: NativeType,
{
    fn backend_any(&self) -> Arc<dyn Any + Send + Sync> {
        self.backend.clone() as Arc<dyn Any + Send + Sync>
    }

    fn from_slice(&self, shape: &[usize], bytes: &[u8]) -> Result<AnyTensor> {
        let typed: &[D] = try_cast_slice(bytes).map_err(|_| Error::TensorTypeMismatch {
            dtype: D::DTYPE,
            device: self.device,
        })?;
        let tensor = Tensor::<B, D>::from_slice(&self.backend, typed, shape)?;
        Ok(AnyTensor::from_tensor(tensor))
    }

    fn zeros(&self, shape: &[usize]) -> Result<AnyTensor> {
        let tensor = Tensor::<B, D>::zeros(&self.backend, shape)?;
        Ok(AnyTensor::from_tensor(tensor))
    }
}

#[derive(Default)]
pub struct BackendRegistry {
    entries: HashMap<(DeviceKind, DType), Arc<dyn ErasedBackendSlot>>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn register<B, D>(&mut self, device: DeviceKind, backend: Arc<B>) -> Result<()>
    where
        B: Backend<D> + 'static,
        D: NativeType,
    {
        let key = (device, D::DTYPE);
        if self.entries.contains_key(&key) {
            return Err(Error::BackendAlreadyRegistered {
                device,
                dtype: D::DTYPE,
            });
        }
        let slot: Arc<dyn ErasedBackendSlot> =
            Arc::new(TypedBackendSlot::<B, D>::new(device, backend));
        self.entries.insert(key, slot);
        Ok(())
    }

    fn entry(&self, device: DeviceKind, dtype: DType) -> Result<&Arc<dyn ErasedBackendSlot>> {
        self.entries
            .get(&(device, dtype))
            .ok_or(Error::BackendNotRegistered { device, dtype })
    }

    pub fn backend<B, D>(&self, device: DeviceKind) -> Result<Arc<B>>
    where
        B: Backend<D> + 'static,
        D: NativeType,
    {
        let entry = self.entry(device, D::DTYPE)?;
        entry
            .backend_any()
            .downcast::<B>()
            .map_err(|_| Error::BackendTypeMismatch {
                device,
                dtype: D::DTYPE,
            })
    }

    fn erased(&self, device: DeviceKind, dtype: DType) -> Result<&Arc<dyn ErasedBackendSlot>> {
        self.entry(device, dtype)
    }
}

pub struct Runtime {
    registry: BackendRegistry,
    default_device: DeviceKind,
}

impl Runtime {
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::new()
    }

    pub fn tensor_from_slice<T: NativeType>(
        &self,
        shape: &[usize],
        data: &[T],
    ) -> Result<AnyTensor> {
        self.tensor_from_slice_on(self.default_device, shape, data)
    }

    pub fn tensor_from_slice_on<T: NativeType>(
        &self,
        device: DeviceKind,
        shape: &[usize],
        data: &[T],
    ) -> Result<AnyTensor> {
        let entry = self.registry.erased(device, T::DTYPE)?;
        let bytes = bytemuck::cast_slice(data);
        entry.from_slice(shape, bytes)
    }

    pub fn tensor_zeros(&self, shape: &[usize], dtype: DType) -> Result<AnyTensor> {
        self.tensor_zeros_on(self.default_device, shape, dtype)
    }

    pub fn tensor_zeros_on(
        &self,
        device: DeviceKind,
        shape: &[usize],
        dtype: DType,
    ) -> Result<AnyTensor> {
        let entry = self.registry.erased(device, dtype)?;
        entry.zeros(shape)
    }

    pub fn tensor<B, D>(&self, shape: &[usize], data: &[D]) -> Result<Tensor<B, D>>
    where
        B: Backend<D> + 'static,
        D: NativeType,
    {
        self.tensor_on(self.default_device, shape, data)
    }

    pub fn tensor_on<B, D>(
        &self,
        device: DeviceKind,
        shape: &[usize],
        data: &[D],
    ) -> Result<Tensor<B, D>>
    where
        B: Backend<D> + 'static,
        D: NativeType,
    {
        let backend = self.backend::<B, D>(device)?;
        Tensor::from_slice(&backend, data, shape)
    }

    pub fn zeros<B, D>(&self, shape: &[usize]) -> Result<Tensor<B, D>>
    where
        B: Backend<D> + 'static,
        D: NativeType,
    {
        self.zeros_on(self.default_device, shape)
    }

    pub fn zeros_on<B, D>(&self, device: DeviceKind, shape: &[usize]) -> Result<Tensor<B, D>>
    where
        B: Backend<D> + 'static,
        D: NativeType,
    {
        let backend = self.backend::<B, D>(device)?;
        Tensor::zeros(&backend, shape)
    }

    pub fn backend<B, D>(&self, device: DeviceKind) -> Result<Arc<B>>
    where
        B: Backend<D> + 'static,
        D: NativeType,
    {
        self.registry.backend(device)
    }
}

pub struct RuntimeBuilder {
    registry: BackendRegistry,
    default_device: Option<DeviceKind>,
}

impl RuntimeBuilder {
    pub fn new() -> Self {
        Self {
            registry: BackendRegistry::new(),
            default_device: None,
        }
    }

    pub fn register_backend<B, D>(mut self, device: DeviceKind, backend: Arc<B>) -> Result<Self>
    where
        B: Backend<D> + 'static,
        D: NativeType,
    {
        self.registry.register(device, backend)?;
        Ok(self)
    }

    pub fn with_default_device(mut self, device: DeviceKind) -> Self {
        self.default_device = Some(device);
        self
    }

    pub fn build(self) -> Result<Runtime> {
        let default_device = self
            .default_device
            .ok_or_else(|| Error::Device("default device not configured".into()))?;
        Ok(Runtime {
            registry: self.registry,
            default_device,
        })
    }
}
