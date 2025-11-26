use std::{mem::MaybeUninit, sync::Arc};
use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use std::alloc::Layout;

#[cfg(feature = "diagnostics")]
use super::allocator::CpuAllocTelemetry;
use super::memory_pool::MemoryPool;

use bolt_core::{
    DType, DeviceKind,
    dtype::NativeType,
    error::{Error, Result},
    layout::Layout as TensorLayout,
    storage::TensorView,
};

#[derive(Debug)]
pub struct StorageBlock<D: NativeType> {
    data: ManuallyDrop<Vec<MaybeUninit<D>>>,
    pool: Option<Arc<MemoryPool>>,
    layout: Layout,
    #[cfg(feature = "diagnostics")]
    diagnostics: Option<Arc<CpuAllocTelemetry>>,
}

impl<D: NativeType> StorageBlock<D> {
    pub(crate) fn new(
        len: usize,
        zeroed: bool,
        pool: Option<Arc<MemoryPool>>,
        #[cfg(feature = "diagnostics")] diagnostics: Option<Arc<CpuAllocTelemetry>>,
    ) -> Result<Self> {
        let (data, layout) = if let Some(pool_ref) = &pool {
            let align = std::mem::align_of::<D>().max(64);
            let size = len.checked_mul(std::mem::size_of::<D>())
                 .ok_or(Error::TensorTooLarge { limit: usize::MAX, requested: usize::MAX })?;
            
            if size == 0 {
                 return Ok(Self {
                     data: ManuallyDrop::new(Vec::new()),
                     pool: None,
                     layout: Layout::new::<D>(),
                     #[cfg(feature = "diagnostics")]
                     diagnostics,
                 });
            }

            let layout = Layout::from_size_align(size, align)
                .map_err(|_| Error::invalid_shape("Invalid memory layout request"))?;

            let ptr = pool_ref.acquire(layout);
            let cap = size / std::mem::size_of::<D>();
            let mut vec = unsafe { 
                Vec::from_raw_parts(ptr.as_ptr() as *mut MaybeUninit<D>, len, cap) 
            };
            
            if zeroed {
               unsafe {
                   std::ptr::write_bytes(vec.as_mut_ptr(), 0, len);
               }
            }
            
            (vec, layout)
        } else {
            let mut vec = Vec::with_capacity(len);
            unsafe { vec.set_len(len) };
            
            if zeroed {
                unsafe {
                   std::ptr::write_bytes(vec.as_mut_ptr(), 0, len);
                }
            }
            
            let layout = Layout::array::<D>(len).unwrap_or(Layout::new::<D>()); 
            (vec, layout)
        };

        Ok(Self {
            data: ManuallyDrop::new(data),
            pool,
            layout,
            #[cfg(feature = "diagnostics")]
            diagnostics,
        })
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

    pub unsafe fn assume_init_slice(&self) -> &[D] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const D, self.data.len()) }
    }

    pub unsafe fn assume_init_slice_mut(&mut self) -> &mut [D] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut D, self.data.len()) }
    }
}

impl<D: NativeType> Drop for StorageBlock<D> {
    fn drop(&mut self) {
        #[cfg(feature = "diagnostics")]
        if let Some(diag) = &self.diagnostics {
            let bytes = self.data.len().saturating_mul(D::DTYPE.size_in_bytes());
            diag.record_dealloc(bytes as u64);
        }

        if let Some(pool) = &self.pool {
             let ptr = self.data.as_mut_ptr() as *mut u8;
             unsafe {
                 if let Some(non_null) = NonNull::new(ptr) {
                     pool.release(non_null, self.layout);
                 }
             }
        } else {
             unsafe { ManuallyDrop::drop(&mut self.data) };
        }
    }
}

#[derive(Clone, Debug)]
pub struct CpuStorage<D: NativeType> {
    block: Arc<StorageBlock<D>>,
}

impl<D: NativeType> CpuStorage<D> {
    pub fn new(block: Arc<StorageBlock<D>>) -> Self {
        Self { block }
    }

    pub fn block(&self) -> &Arc<StorageBlock<D>> {
        &self.block
    }

    pub fn len_bytes(&self) -> usize {
        self.block.len() * D::DTYPE.size_in_bytes()
    }

    pub fn as_uninit_slice(&self) -> &[MaybeUninit<D>] {
        self.block.as_uninit_slice()
    }

    pub unsafe fn as_slice(&self) -> &[D] {
        unsafe { self.block.assume_init_slice() }
    }

    pub unsafe fn try_as_mut_slice(&mut self) -> Result<&mut [D]> {
        let block = Arc::get_mut(&mut self.block).ok_or_else(|| {
            Error::OpError("cannot write to shared tensor storage; clone before mutating".into())
        })?;
        Ok(unsafe { block.assume_init_slice_mut() })
    }

    pub fn try_as_uninit_slice_mut(&mut self) -> Result<&mut [MaybeUninit<D>]> {
        let block = Arc::get_mut(&mut self.block).ok_or_else(|| {
            Error::OpError("cannot write to shared tensor storage; clone before mutating".into())
        })?;
        Ok(block.as_uninit_slice_mut())
    }

    pub fn dtype(&self) -> DType {
        D::DTYPE
    }
    
    pub fn device_kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }
}

pub unsafe fn read_into_slice<D: NativeType>(
    storage: &CpuStorage<D>,
    layout: &TensorLayout,
    dst: &mut [D],
) -> Result<()> {
    if layout.num_elements() != dst.len() {
        return Err(Error::SizeMismatch {
            expected: layout.num_elements(),
            actual: dst.len(),
        });
    }
    let uninit_dst: &mut [MaybeUninit<D>] = unsafe {
        std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut MaybeUninit<D>, dst.len())
    };
    unsafe { read_into_uninit_slice(storage, layout, uninit_dst) }
}

pub unsafe fn read_into_uninit_slice<D: NativeType>(
    storage: &CpuStorage<D>,
    layout: &TensorLayout,
    dst: &mut [MaybeUninit<D>],
) -> Result<()> {
    if layout.num_elements() != dst.len() {
        return Err(Error::SizeMismatch {
            expected: layout.num_elements(),
            actual: dst.len(),
        });
    }
    
    layout.validate_bounds(D::DTYPE, storage.len_bytes())?;

    let data = storage.block.as_uninit_slice();
    if layout.is_contiguous() && layout.offset_bytes() == 0 {
        let len = dst.len();
        let init = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const D, len) };
        for (slot, value) in dst.iter_mut().take(len).zip(init.iter().copied()) {
            slot.write(value);
        }
        return Ok(());
    }
    let elem_size = D::DTYPE.size_in_bytes();
    let iter = layout.iter_offsets(D::DTYPE)?;
    for (idx, src_bytes) in iter.enumerate() {
        debug_assert_eq!(src_bytes % elem_size, 0);
        let src = src_bytes / elem_size;
        let value = unsafe { data[src].assume_init() };
        dst[idx].write(value);
    }
    Ok(())
}

pub fn write_from_slice<D: NativeType>(
    storage: &mut CpuStorage<D>,
    layout: &TensorLayout,
    src: &[D],
) -> Result<()> {
    if layout.num_elements() != src.len() {
        return Err(Error::SizeMismatch {
            expected: layout.num_elements(),
            actual: src.len(),
        });
    }

    layout.validate_bounds(D::DTYPE, storage.len_bytes())?;

    let dst = storage.try_as_uninit_slice_mut()?;
    if layout.is_contiguous() && layout.offset_bytes() == 0 {
        for (slot, value) in dst.iter_mut().take(src.len()).zip(src.iter().copied()) {
            slot.write(value);
        }
        return Ok(());
    }
    let elem_size = D::DTYPE.size_in_bytes();
    let iter = layout.iter_offsets(D::DTYPE)?;
    for (idx, dst_bytes) in iter.enumerate() {
        debug_assert_eq!(dst_bytes % elem_size, 0);
        let dst_idx = dst_bytes / elem_size;
        dst[dst_idx].write(src[idx]);
    }
    Ok(())
}

pub fn fill_storage<D: NativeType>(
    storage: &mut CpuStorage<D>,
    layout: &TensorLayout,
    value: D,
) -> Result<()> {
    layout.validate_bounds(D::DTYPE, storage.len_bytes())?;
    let dst = storage.try_as_uninit_slice_mut()?;
    if layout.is_contiguous() && layout.offset_bytes() == 0 {
        let len = layout.num_elements();
        for slot in dst.iter_mut().take(len) {
            slot.write(value);
        }
        return Ok(());
    }
    let elem_size = D::DTYPE.size_in_bytes();
    let iter = layout.iter_offsets(D::DTYPE)?;
    for dst_bytes in iter {
        debug_assert_eq!(dst_bytes % elem_size, 0);
        let dst_idx = dst_bytes / elem_size;
        dst[dst_idx].write(value);
    }
    Ok(())
}

pub type CpuTensorView<'a, D> = TensorView<'a, CpuStorage<D>>;
