use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use bolt_core::backend::{Backend, TensorParts, AddOp, SubOp, MatmulOp, MeanOp, CopyOp, FillOp};
use bolt_core::dtype::NativeType;
use bolt_core::error::Result;
use bolt_core::layout::Layout;

use crate::profile;
use crate::report::ProfileReport;
use crate::allocator::TrackingAllocator;

/// A registry to store aggregated profiling stats per operation name.
#[derive(Default, Debug)]
pub struct Registry {
    /// Map of Op Name -> List of reports
    ops: HashMap<String, Vec<ProfileReport>>,
}

impl Registry {
    pub fn add(&mut self, name: &str, report: ProfileReport) {
        self.ops.entry(name.to_string()).or_default().push(report);
    }

    pub fn print_summary(&self) {
        println!("\n=== Profiling Summary ===");
        println!("| {:<20} | {:<5} | {:<12} | {:<12} | {:<12} |", 
            "Op Name", "Count", "Avg Time(µs)", "Avg Net(B)", "Avg Peak(B)");
        println!("|{:-<22}|{:-<7}|{:-<14}|{:-<14}|{:-<14}|", "", "", "", "", "");

        for (name, reports) in &self.ops {
            let count = reports.len();
            if count == 0 { continue; }
            
            let total_time: u128 = reports.iter().map(|r| r.wall_time.as_micros()).sum();
            let total_net: isize = reports.iter().map(|r| r.memory_stats.net_allocated_bytes).sum();
            let max_rss: u64 = reports.iter().map(|r| r.memory_stats.peak_rss_bytes).max().unwrap_or(0);

            println!("| {:<20} | {:<5} | {:<12} | {:<12} | {:<12} |",
                name,
                count,
                total_time / count as u128,
                total_net / count as isize,
                max_rss
            );
        }
        println!("=========================\n");
    }
}

/// A Backend Decorator that profiles every operation.
#[derive(Debug, Clone)]
pub struct ProfiledBackend<B> {
    inner: B,
    registry: Arc<Mutex<Registry>>,
    allocator: Option<&'static TrackingAllocator>,
}

impl<B> ProfiledBackend<B> {
    /// Creates a new ProfiledBackend wrapping the given backend.
    ///
    /// # Arguments
    /// * `inner` - The backend to wrap (e.g. CpuBackend).
    /// * `allocator` - Optional global allocator tracker for precise memory stats.
    pub fn new(inner: B, allocator: Option<&'static TrackingAllocator>) -> Self {
        Self {
            inner,
            registry: Arc::new(Mutex::new(Registry::default())),
            allocator,
        }
    }

    /// Prints the aggregated report to stdout.
    pub fn print_report(&self) {
        let reg = self.registry.lock().unwrap();
        reg.print_summary();
    }

    fn profile_op<F, R>(&self, name: &str, op: F) -> R 
    where F: FnOnce() -> R {
        let (result, report) = profile(self.allocator, op);
        self.registry.lock().unwrap().add(name, report);
        result
    }
}

impl<D: NativeType, B: Backend<D>> Backend<D> for ProfiledBackend<B> {
    type Device = B::Device;
    type Storage = B::Storage;
    type Allocator = B::Allocator;

    fn device(&self) -> &Self::Device {
        self.inner.device()
    }

    fn allocator(&self) -> Self::Allocator {
        self.inner.allocator()
    }

    fn storage_len_bytes(&self, storage: &Self::Storage) -> usize {
        self.inner.storage_len_bytes(storage)
    }

    fn read(&self, storage: &Self::Storage, layout: &Layout, dst: &mut [D]) -> Result<()> {
        self.profile_op("read", || self.inner.read(storage, layout, dst))
    }

    fn write(&self, storage: &mut Self::Storage, layout: &Layout, src: &[D]) -> Result<()> {
        self.profile_op("write", || self.inner.write(storage, layout, src))
    }
}

impl<D: NativeType, B: CopyOp<D>> CopyOp<D> for ProfiledBackend<B> {
    fn copy(&self, storage: &Self::Storage, layout: &Layout) -> Result<TensorParts<Self::Storage>> {
        self.profile_op("copy", || self.inner.copy(storage, layout))
    }
}

impl<D: NativeType, B: FillOp<D>> FillOp<D> for ProfiledBackend<B> {
    fn fill(&self, layout: &Layout, value: D) -> Result<Self::Storage> {
        self.profile_op("fill", || self.inner.fill(layout, value))
    }
}

impl<D: NativeType, B: AddOp<D>> AddOp<D> for ProfiledBackend<B> {
    fn add(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage>> {
        self.profile_op("add", || self.inner.add(lhs, rhs, lhs_layout, rhs_layout))
    }
}

impl<D: NativeType, B: SubOp<D>> SubOp<D> for ProfiledBackend<B> {
    fn sub(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage>> {
        self.profile_op("sub", || self.inner.sub(lhs, rhs, lhs_layout, rhs_layout))
    }
}

impl<D: NativeType, B: MatmulOp<D>> MatmulOp<D> for ProfiledBackend<B> {
    fn matmul(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<TensorParts<Self::Storage>> {
        self.profile_op("matmul", || self.inner.matmul(lhs, rhs, lhs_layout, rhs_layout))
    }
}

impl<D: NativeType, B: MeanOp<D>> MeanOp<D> for ProfiledBackend<B> 
where 
    B: Backend<f32>,
    ProfiledBackend<B>: Backend<f32>
{
    fn mean_f32(
        &self,
        storage: &<Self as Backend<D>>::Storage,
        layout: &Layout,
    ) -> Result<TensorParts<<Self as Backend<f32>>::Storage>> {
        let (res, report) = profile(self.allocator, || self.inner.mean_f32(storage, layout));
        self.registry.lock().unwrap().add("mean_f32", report);
        
        let parts = res?;
        // We manually construct the return type to help inference
        let storage: <B as Backend<f32>>::Storage = parts.storage;
        
        // Force the compiler to accept that B::Storage == ProfiledBackend<B>::Storage.
        // They are defined as equal in the impl block, but generic bounds confuse the checker.
        let storage_ptr = &storage as *const <B as Backend<f32>>::Storage;
        let casted_storage = unsafe {
             std::ptr::read(storage_ptr as *const <Self as Backend<f32>>::Storage)
        };
        // Forget the original to avoid double drop
        std::mem::forget(storage);
        
        Ok(TensorParts {
            storage: casted_storage,
            layout: parts.layout,
        })
    }
}