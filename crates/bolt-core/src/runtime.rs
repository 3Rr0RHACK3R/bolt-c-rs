use std::{any::Any, collections::HashMap, sync::Arc};

use bytemuck::try_cast_slice;

use crate::{
    any_tensor::AnyTensor,
    backend::Backend,
    device::DeviceKind,
    dtype::{DType, NativeType},
    error::{Error, Result},
    tensor::Tensor,
};

type SliceFn = dyn Fn(&[usize], &[u8]) -> Result<AnyTensor> + Send + Sync;
type ZerosFn = dyn Fn(&[usize]) -> Result<AnyTensor> + Send + Sync;

struct BackendRecord {
    backend: Arc<dyn Any + Send + Sync>,
    from_slice: Arc<SliceFn>,
    zeros: Arc<ZerosFn>,
}

#[derive(Default)]
pub struct BackendRegistry {
    entries: HashMap<(DeviceKind, DType), BackendRecord>,
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
        let backend_clone = backend.clone();
        let from_slice = Arc::new(move |shape: &[usize], bytes: &[u8]| -> Result<AnyTensor> {
            let typed: &[D] = try_cast_slice(bytes).map_err(|_| Error::TensorTypeMismatch {
                dtype: D::DTYPE,
                device,
            })?;
            let tensor = Tensor::<B, D>::from_slice(&backend_clone, typed, shape)?;
            Ok(AnyTensor::from_tensor(tensor))
        });
        let backend_clone = backend.clone();
        let zeros = Arc::new(move |shape: &[usize]| -> Result<AnyTensor> {
            let tensor = Tensor::<B, D>::zeros(&backend_clone, shape)?;
            Ok(AnyTensor::from_tensor(tensor))
        });
        let erased_backend: Arc<dyn Any + Send + Sync> = backend;
        self.entries.insert(
            key,
            BackendRecord {
                backend: erased_backend,
                from_slice,
                zeros,
            },
        );
        Ok(())
    }

    fn entry(&self, device: DeviceKind, dtype: DType) -> Result<&BackendRecord> {
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
            .backend
            .clone()
            .downcast::<B>()
            .map_err(|_| Error::BackendTypeMismatch {
                device,
                dtype: D::DTYPE,
            })
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
        let entry = self.registry.entry(device, T::DTYPE)?;
        let bytes = bytemuck::cast_slice(data);
        (entry.from_slice)(shape, bytes)
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
        let entry = self.registry.entry(device, dtype)?;
        (entry.zeros)(shape)
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
