use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicU64, Ordering},
    },
};

use bolt_core::{
    buffer::BufferId,
    device::{Device, DeviceKind},
    error::{Error, Result},
};

pub struct CpuDevice {
    buffers: Mutex<HashMap<u64, Arc<BufferCell>>>,
    next_id: AtomicU64,
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuDevice {
    pub fn new() -> Self {
        Self {
            buffers: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(0),
        }
    }

    pub(crate) fn buffer_cell(&self, id: BufferId) -> Result<Arc<BufferCell>> {
        let buffers = self.buffers.lock().unwrap();
        buffers
            .get(&id.raw())
            .cloned()
            .ok_or_else(|| Error::Device(format!("invalid buffer id {}", id.raw())))
    }

    fn insert_buffer(&self, size: usize) -> Arc<BufferCell> {
        Arc::new(BufferCell::new(size))
    }
}

impl Device for CpuDevice {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    fn alloc(&self, size: usize, _align: usize) -> Result<BufferId> {
        if size == 0 {
            return Err(Error::invalid_shape("cannot allocate zero bytes"));
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let buffer = self.insert_buffer(size);
        let mut buffers = self.buffers.lock().unwrap();
        buffers.insert(id, buffer);
        Ok(BufferId::new(id))
    }

    fn free(&self, buffer: BufferId) -> Result<()> {
        let mut buffers = self.buffers.lock().unwrap();
        buffers
            .remove(&buffer.raw())
            .ok_or_else(|| Error::Device(format!("double free buffer {}", buffer.raw())))
            .map(|_| ())
    }

    fn write(&self, buffer: BufferId, offset: usize, data: &[u8]) -> Result<()> {
        let cell = self.buffer_cell(buffer)?;
        cell.with_write(|dst| {
            let end = offset + data.len();
            if end > dst.len() {
                return Err(Error::SizeMismatch {
                    expected: dst.len() - offset,
                    actual: data.len(),
                });
            }
            dst[offset..end].copy_from_slice(data);
            Ok(())
        })
    }

    fn read(&self, buffer: BufferId, offset: usize, dst: &mut [u8]) -> Result<()> {
        let cell = self.buffer_cell(buffer)?;
        cell.with_read(|src| {
            let end = offset + dst.len();
            if end > src.len() {
                return Err(Error::SizeMismatch {
                    expected: src.len() - offset,
                    actual: dst.len(),
                });
            }
            dst.copy_from_slice(&src[offset..end]);
            Ok(())
        })
    }

    fn copy(
        &self,
        src: BufferId,
        src_offset: usize,
        dst: BufferId,
        dst_offset: usize,
        size: usize,
    ) -> Result<()> {
        if src == dst {
            let cell = self.buffer_cell(src)?;
            return cell.with_write(|data| {
                Self::check_bounds(data.len(), src_offset, size)?;
                Self::check_bounds(data.len(), dst_offset, size)?;
                data.copy_within(src_offset..src_offset + size, dst_offset);
                Ok(())
            });
        }

        // Two-phase copy to avoid nested lock scopes and deadlocks when multiple
        // threads copy opposite directions. Temporary allocation is acceptable
        // today; revisit with lock ordering once profiling shows Device::copy is hot.
        let temp = {
            let cell = self.buffer_cell(src)?;
            cell.with_read(|src_data| {
                Self::check_bounds(src_data.len(), src_offset, size)?;
                Ok(src_data[src_offset..src_offset + size].to_vec())
            })?
        };

        let cell = self.buffer_cell(dst)?;
        cell.with_write(|dst_data| {
            Self::check_bounds(dst_data.len(), dst_offset, size)?;
            dst_data[dst_offset..dst_offset + size].copy_from_slice(&temp);
            Ok(())
        })
    }

    fn sync(&self) -> Result<()> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CpuDevice {
    fn check_bounds(len: usize, offset: usize, size: usize) -> Result<()> {
        if offset.checked_add(size).is_none_or(|end| end > len) {
            return Err(Error::SizeMismatch {
                expected: len.saturating_sub(offset),
                actual: size,
            });
        }
        Ok(())
    }
}

pub(crate) struct BufferCell {
    data: RwLock<Vec<u8>>,
}

impl BufferCell {
    fn new(size: usize) -> Self {
        Self {
            data: RwLock::new(vec![0u8; size]),
        }
    }

    pub(crate) fn with_write<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&mut [u8]) -> Result<T>,
    {
        let mut guard = self.data.write().unwrap();
        f(&mut guard)
    }

    pub(crate) fn with_read<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&[u8]) -> Result<T>,
    {
        let guard = self.data.read().unwrap();
        f(&guard)
    }
}
