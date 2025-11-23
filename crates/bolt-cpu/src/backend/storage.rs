use std::{mem::MaybeUninit, sync::Arc};

use bolt_core::{
    DType, DeviceKind,
    dtype::NativeType,
    error::{Error, Result},
    layout::Layout,
    storage::{BufferHandle, TensorView},
};

use super::layout_utils::{expand_strides, linear_to_indices, offset_from_strides};

#[derive(Debug)]
pub struct StorageBlock<D: NativeType> {
    data: Vec<MaybeUninit<D>>,
}

impl<D: NativeType> StorageBlock<D> {
    pub fn new(len: usize, zeroed: bool) -> Self {
        let mut data = if zeroed {
            vec![MaybeUninit::new(D::default()); len]
        } else {
            let mut vec = Vec::with_capacity(len);
            unsafe { vec.set_len(len) };
            vec
        };
        if len == 0 {
            data.clear();
        }
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn as_uninit_slice(&self) -> &[MaybeUninit<D>] {
        &self.data
    }

    pub fn as_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<D>] {
        &mut self.data
    }

    pub fn assume_init_slice(&self) -> &[D] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const D, self.data.len()) }
    }

    pub fn assume_init_slice_mut(&mut self) -> &mut [D] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut D, self.data.len()) }
    }
}

#[derive(Clone, Debug)]
pub struct CpuStorage<D: NativeType> {
    handle: BufferHandle,
    block: Arc<StorageBlock<D>>,
}

impl<D: NativeType> CpuStorage<D> {
    pub fn new(handle: BufferHandle, block: Arc<StorageBlock<D>>) -> Self {
        Self { handle, block }
    }

    pub fn handle(&self) -> &BufferHandle {
        &self.handle
    }

    pub fn block(&self) -> &Arc<StorageBlock<D>> {
        &self.block
    }

    pub fn len_bytes(&self) -> usize {
        self.handle.len_bytes()
    }

    /// Returns a typed slice over the storage.
    /// Precondition: the entire buffer must be fully initialized. Prefer
    /// `as_uninit_slice` when initialization is partial or unknown.
    pub fn as_uninit_slice(&self) -> &[MaybeUninit<D>] {
        self.block.as_uninit_slice()
    }

    /// Returns a typed slice over the storage.
    /// Precondition: the entire buffer must be fully initialized. Prefer
    /// `as_uninit_slice` when initialization is partial or unknown.
    pub fn as_slice(&self) -> &[D] {
        self.block.assume_init_slice()
    }

    /// Returns a mutable typed slice over the storage.
    /// Precondition: the entire buffer must be fully initialized. Prefer
    /// `try_as_uninit_slice_mut` when initialization is partial or unknown.
    pub fn try_as_mut_slice(&mut self) -> Result<&mut [D]> {
        let block = Arc::get_mut(&mut self.block).ok_or_else(|| {
            Error::OpError("cannot write to shared tensor storage; clone before mutating".into())
        })?;
        Ok(block.assume_init_slice_mut())
    }

    pub fn try_as_uninit_slice_mut(&mut self) -> Result<&mut [MaybeUninit<D>]> {
        let block = Arc::get_mut(&mut self.block).ok_or_else(|| {
            Error::OpError("cannot write to shared tensor storage; clone before mutating".into())
        })?;
        Ok(block.as_uninit_slice_mut())
    }

    pub fn dtype(&self) -> DType {
        self.handle.dtype()
    }
}

pub fn read_into_slice<D: NativeType>(
    storage: &CpuStorage<D>,
    layout: &Layout,
    dst: &mut [D],
) -> Result<()> {
    if layout.num_elements() != dst.len() {
        return Err(Error::SizeMismatch {
            expected: layout.num_elements(),
            actual: dst.len(),
        });
    }
    // Safety: casting an initialized slice to MaybeUninit preserves layout;
    // we immediately write all elements through the MaybeUninit view.
    let uninit_dst: &mut [MaybeUninit<D>] = unsafe { std::mem::transmute(dst) };
    read_into_uninit_slice(storage, layout, uninit_dst)
}

pub fn read_into_uninit_slice<D: NativeType>(
    storage: &CpuStorage<D>,
    layout: &Layout,
    dst: &mut [MaybeUninit<D>],
) -> Result<()> {
    if layout.num_elements() != dst.len() {
        return Err(Error::SizeMismatch {
            expected: layout.num_elements(),
            actual: dst.len(),
        });
    }
    storage.handle().validate_layout(layout)?;
    let data = storage.block.as_uninit_slice();
    if layout.is_contiguous() && layout.offset_bytes() == 0 {
        let len = dst.len();
        let init = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const D, len) };
        for (slot, value) in dst.iter_mut().take(len).zip(init.iter().copied()) {
            slot.write(value);
        }
        return Ok(());
    }
    let shape = layout.shape();
    let strides = expand_strides(layout, shape)?;
    let offset = layout.offset_elements(D::DTYPE);
    let mut coords = vec![0usize; shape.len()];
    for idx in 0..layout.num_elements() {
        linear_to_indices(idx, shape, &mut coords);
        let src = offset + offset_from_strides(&coords, &strides);
        let value = unsafe { data[src as usize].assume_init() };
        dst[idx].write(value);
    }
    Ok(())
}

pub fn write_from_slice<D: NativeType>(
    storage: &mut CpuStorage<D>,
    layout: &Layout,
    src: &[D],
) -> Result<()> {
    if layout.num_elements() != src.len() {
        return Err(Error::SizeMismatch {
            expected: layout.num_elements(),
            actual: src.len(),
        });
    }
    storage.handle().validate_layout(layout)?;
    let dst = storage.try_as_uninit_slice_mut()?;
    if layout.is_contiguous() && layout.offset_bytes() == 0 {
        for (slot, value) in dst.iter_mut().take(src.len()).zip(src.iter().copied()) {
            slot.write(value);
        }
        return Ok(());
    }
    let shape = layout.shape();
    let strides = expand_strides(layout, shape)?;
    let offset = layout.offset_elements(D::DTYPE);
    let mut coords = vec![0usize; shape.len()];
    for idx in 0..layout.num_elements() {
        linear_to_indices(idx, shape, &mut coords);
        let dst_idx = offset + offset_from_strides(&coords, &strides);
        dst[dst_idx as usize].write(src[idx]);
    }
    Ok(())
}

pub fn fill_storage<D: NativeType>(
    storage: &mut CpuStorage<D>,
    layout: &Layout,
    value: D,
) -> Result<()> {
    storage.handle().validate_layout(layout)?;
    let dst = storage.try_as_uninit_slice_mut()?;
    if layout.is_contiguous() && layout.offset_bytes() == 0 {
        let len = layout.num_elements();
        for slot in dst.iter_mut().take(len) {
            slot.write(value);
        }
        return Ok(());
    }
    let shape = layout.shape();
    let strides = expand_strides(layout, shape)?;
    let offset = layout.offset_elements(D::DTYPE);
    let mut coords = vec![0usize; shape.len()];
    for idx in 0..layout.num_elements() {
        linear_to_indices(idx, shape, &mut coords);
        let dst_idx = offset + offset_from_strides(&coords, &strides);
        dst[dst_idx as usize].write(value);
    }
    Ok(())
}

pub fn make_cpu_handle<D: NativeType>(len: usize) -> Result<BufferHandle> {
    let bytes = len
        .checked_mul(D::DTYPE.size_in_bytes())
        .ok_or_else(|| Error::TensorTooLarge {
            limit: usize::MAX,
            requested: usize::MAX,
        })?;
    BufferHandle::new(DeviceKind::Cpu, D::DTYPE, bytes)
}

pub type CpuTensorView<'a, D> = TensorView<'a, CpuStorage<D>>;
