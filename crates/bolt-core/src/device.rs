use std::any::Any;

use crate::{buffer::BufferId, error::Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    Cuda,
}

pub trait Device: Send + Sync {
    fn kind(&self) -> DeviceKind;

    fn alloc(&self, size: usize, align: usize) -> Result<BufferId>;

    fn free(&self, buffer: BufferId) -> Result<()>;

    fn write(&self, buffer: BufferId, offset: usize, data: &[u8]) -> Result<()>;

    fn read(&self, buffer: BufferId, offset: usize, dst: &mut [u8]) -> Result<()>;

    fn copy(
        &self,
        src: BufferId,
        src_offset: usize,
        dst: BufferId,
        dst_offset: usize,
        size: usize,
    ) -> Result<()>;

    fn sync(&self) -> Result<()>;

    fn as_any(&self) -> &dyn Any;
}
