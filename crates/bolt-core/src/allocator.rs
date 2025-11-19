use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use crate::{dtype::NativeType, error::Result};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AllocatorMetrics {
    pub bytes_reserved: u64,
    pub bytes_in_use: u64,
    pub alloc_failures: u64,
}

#[derive(Debug, Default)]
struct AllocatorCounters {
    bytes_reserved: AtomicU64,
    bytes_in_use: AtomicU64,
    alloc_failures: AtomicU64,
}

impl AllocatorCounters {
    fn snapshot(&self) -> AllocatorMetrics {
        AllocatorMetrics {
            bytes_reserved: self.bytes_reserved.load(Ordering::Relaxed),
            bytes_in_use: self.bytes_in_use.load(Ordering::Relaxed),
            alloc_failures: self.alloc_failures.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AllocatorHandle {
    counters: Arc<AllocatorCounters>,
}

impl AllocatorHandle {
    pub fn new() -> Self {
        Self {
            counters: Arc::new(AllocatorCounters::default()),
        }
    }

    pub fn snapshot(&self) -> AllocatorMetrics {
        self.counters.snapshot()
    }

    fn counters(&self) -> Arc<AllocatorCounters> {
        self.counters.clone()
    }

    fn record_alloc<D: NativeType>(&self, len: usize) {
        let bytes = (len * D::DTYPE.size_in_bytes()) as u64;
        self.counters
            .bytes_reserved
            .fetch_add(bytes, Ordering::Relaxed);
        self.counters
            .bytes_in_use
            .fetch_add(bytes, Ordering::Relaxed);
    }

}

pub struct StorageBlock<D: NativeType> {
    data: Vec<D>,
    counters: Arc<AllocatorCounters>,
}

impl<D: NativeType> StorageBlock<D> {
    pub fn new(len: usize, handle: &AllocatorHandle, zeroed: bool) -> Self {
        handle.record_alloc::<D>(len);
        let counters = handle.counters();
        let mut data = if zeroed {
            vec![D::default(); len]
        } else {
            let mut vec = Vec::with_capacity(len);
            unsafe { vec.set_len(len) };
            vec
        };
        if len == 0 {
            data.clear();
        }
        Self { data, counters }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn as_slice(&self) -> &[D] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [D] {
        &mut self.data
    }
}

impl<D: NativeType> Drop for StorageBlock<D> {
    fn drop(&mut self) {
        let bytes = (self.data.len() * D::DTYPE.size_in_bytes()) as u64;
        self.counters
            .bytes_in_use
            .fetch_sub(bytes, Ordering::Relaxed);
    }
}

pub trait StorageAllocator<D: NativeType>: Send + Sync {
    type Storage: Clone + Send + Sync + 'static;

    fn allocate(&self, len: usize) -> Result<Self::Storage>;
    fn allocate_zeroed(&self, len: usize) -> Result<Self::Storage>;
    fn metrics(&self) -> AllocatorMetrics;
}
