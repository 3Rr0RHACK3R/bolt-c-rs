use std::{marker::PhantomData, sync::Arc};

use bolt_core::{
    allocator::StorageAllocator,
    backend::{Backend, TensorParts},
    device::{BackendDevice, DeviceKind},
    dtype::NativeType,
    error::{Error, Result},
    layout::Layout,
    tensor::Tensor,
};

#[derive(Clone, Copy)]
struct TestDevice;

impl BackendDevice for TestDevice {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }
}

#[derive(Clone, Copy)]
struct TestAllocator;

impl<D: NativeType> StorageAllocator<D> for TestAllocator {
    type Storage = Vec<D>;

    fn allocate(&self, len: usize) -> Result<Self::Storage> {
        Ok(vec![D::default(); len])
    }

    fn allocate_zeroed(&self, len: usize) -> Result<Self::Storage> {
        Ok(vec![D::default(); len])
    }
}

#[derive(Clone)]
struct TestBackend<D: NativeType> {
    device: TestDevice,
    allocator: TestAllocator,
    _marker: PhantomData<D>,
}

impl<D: NativeType> TestBackend<D> {
    fn new() -> Self {
        Self {
            device: TestDevice,
            allocator: TestAllocator,
            _marker: PhantomData,
        }
    }
}

impl<D: NativeType> Backend<D> for TestBackend<D> {
    type Device = TestDevice;
    type Storage = Vec<D>;
    type Allocator = TestAllocator;

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn allocator(&self) -> Self::Allocator {
        self.allocator
    }

    fn storage_len_bytes(&self, storage: &Self::Storage) -> usize {
        storage.len() * D::DTYPE.size_in_bytes()
    }

    fn read(&self, storage: &Self::Storage, layout: &Layout, dst: &mut [D]) -> Result<()> {
        layout.validate_bounds(D::DTYPE, self.storage_len_bytes(storage))?;
        copy_with_layout::<D, _, _, _>(layout, storage, dst, |src| *src)
    }

    fn write(&self, storage: &mut Self::Storage, layout: &Layout, src: &[D]) -> Result<()> {
        if src.len() != layout.num_elements() {
            return Err(Error::SizeMismatch {
                expected: layout.num_elements(),
                actual: src.len(),
            });
        }
        layout.validate_bounds(D::DTYPE, self.storage_len_bytes(storage))?;
        copy_with_layout::<D, _, _, _>(layout, src, storage, |value| *value)
    }

    fn copy(&self, storage: &Self::Storage, layout: &Layout) -> Result<TensorParts<Self::Storage>> {
        layout.validate_bounds(D::DTYPE, self.storage_len_bytes(storage))?;
        let mut values = vec![D::default(); layout.num_elements()];
        self.read(storage, layout, &mut values)?;
        let layout = Layout::contiguous(layout.concrete_shape().clone());
        Ok(TensorParts {
            storage: values,
            layout,
        })
    }

    fn add(
        &self,
        _lhs: &Self::Storage,
        _rhs: &Self::Storage,
        _lhs_layout: &Layout,
        _rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage>> {
        Err(Error::OpError("add not implemented in test backend".into()))
    }

    fn sub(
        &self,
        _lhs: &Self::Storage,
        _rhs: &Self::Storage,
        _lhs_layout: &Layout,
        _rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage>> {
        Err(Error::OpError("sub not implemented in test backend".into()))
    }

    fn matmul(
        &self,
        _lhs: &Self::Storage,
        _rhs: &Self::Storage,
        _lhs_layout: &Layout,
        _rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage>> {
        Err(Error::OpError(
            "matmul not implemented in test backend".into(),
        ))
    }

    fn mean_f32(
        &self,
        _storage: &<Self as Backend<D>>::Storage,
        _layout: &Layout,
    ) -> Result<TensorParts<<Self as Backend<f32>>::Storage>>
    where
        Self: Backend<f32>,
    {
        Err(Error::OpError(
            "mean_f32 not implemented in test backend".into(),
        ))
    }
}

fn copy_with_layout<D, S, T, F>(layout: &Layout, src: S, mut dst: T, mut map: F) -> Result<()>
where
    D: NativeType,
    S: AsRef<[D]>,
    T: AsMut<[D]>,
    F: FnMut(&D) -> D,
{
    let src = src.as_ref();
    let dst = dst.as_mut();
    for_each_offset::<D, _>(layout, |linear, offset| {
        dst[linear] = map(&src[offset]);
        Ok(())
    })
}

fn for_each_offset<D, F>(layout: &Layout, mut f: F) -> Result<()>
where
    D: NativeType,
    F: FnMut(usize, usize) -> Result<()>,
{
    let shape = layout.shape();
    let strides = layout.strides();
    let base = layout.offset_elements(D::DTYPE);
    if base < 0 {
        return Err(Error::invalid_shape("negative base offset"));
    }
    let base = base as isize;
    for linear in 0..layout.num_elements() {
        let mut rem = linear;
        let mut offset = base;
        for (dim, stride) in shape.iter().zip(strides.iter()) {
            let idx = rem % *dim;
            rem /= *dim;
            offset += (idx as isize) * *stride;
        }
        if offset < 0 {
            return Err(Error::invalid_shape("negative offset"));
        }
        f(linear, offset as usize)?;
    }
    Ok(())
}

fn backend<D: NativeType>() -> Arc<TestBackend<D>> {
    Arc::new(TestBackend::new())
}

#[test]
fn item_returns_scalar_value() -> Result<()> {
    let backend = backend::<f32>();
    let tensor = Tensor::from_slice(&backend, &[42.0f32], &[1])?;

    let value = tensor.item()?;

    assert_eq!(value, 42.0);
    Ok(())
}

#[test]
fn item_rejects_multi_element_tensor() -> Result<()> {
    let backend = backend::<i32>();
    let tensor = Tensor::from_slice(&backend, &[1i32, 2], &[2])?;

    let err = tensor.item();

    assert!(matches!(err, Err(Error::InvalidShape { .. })));
    Ok(())
}

#[test]
fn item_preserves_dtype_for_i32() -> Result<()> {
    let backend = backend::<i32>();
    let tensor = Tensor::from_slice(&backend, &[7i32], &[1])?;

    let value = tensor.item()?;

    assert_eq!(value, 7i32);
    Ok(())
}

#[test]
fn item_handles_strided_scalar_views() -> Result<()> {
    let backend = backend::<f32>();
    let tensor = Tensor::from_slice(&backend, &[1.0f32, 2.0, 3.0, 4.0], &[4])?;
    let every_other = tensor.slice(0, 1, 4, 2)?; // [2.0, 4.0]
    let last = every_other.slice(0, 1, 2, 1)?; // [4.0]

    let value = last.item()?;

    assert_eq!(value, 4.0);
    Ok(())
}
