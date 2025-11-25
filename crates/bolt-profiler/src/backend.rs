use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use bolt_core::backend::{AddOp, Backend, CopyOp, FillOp, MatmulOp, MeanOp, SubOp, TensorParts};
use bolt_core::dtype::NativeType;
use bolt_core::error::Result;
use bolt_core::layout::Layout;
use parking_lot::Mutex;

use crate::allocator::TrackingAllocator;
use crate::profile;
use crate::report::ProfileReport;

#[derive(Debug, Clone, Default)]
pub struct OpStats {
    pub count: usize,
    pub total_time_us: u128,
    pub min_time_us: u128,
    pub max_time_us: u128,
    pub sum_sq_time_us: u128,
    pub total_net_bytes: isize,
    pub total_alloc_bytes: usize,
    pub max_scope_peak_bytes: usize,
}

impl OpStats {
    pub fn record(&mut self, report: &ProfileReport) {
        let time_us = report.wall_time.as_micros();
        self.count += 1;
        self.total_time_us += time_us;
        self.min_time_us = if self.count == 1 {
            time_us
        } else {
            self.min_time_us.min(time_us)
        };
        self.max_time_us = self.max_time_us.max(time_us);
        self.sum_sq_time_us += time_us * time_us;
        self.total_net_bytes += report.memory_stats.net_allocated_bytes;
        self.total_alloc_bytes += report.memory_stats.total_allocated_bytes;
        self.max_scope_peak_bytes = self
            .max_scope_peak_bytes
            .max(report.memory_stats.scope_peak_bytes);
    }

    pub fn avg_time_us(&self) -> u128 {
        if self.count == 0 {
            0
        } else {
            self.total_time_us / self.count as u128
        }
    }

    pub fn stddev_time_us(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let n = self.count as f64;
        let mean = self.total_time_us as f64 / n;
        let variance = (self.sum_sq_time_us as f64 / n) - (mean * mean);
        variance.max(0.0).sqrt()
    }

    pub fn avg_net_bytes(&self) -> isize {
        if self.count == 0 {
            0
        } else {
            self.total_net_bytes / self.count as isize
        }
    }
}

#[derive(Default, Debug)]
pub struct Registry {
    ops: BTreeMap<String, OpStats>,
    total_time: Duration,
}

impl Registry {
    pub fn add(&mut self, name: &str, report: &ProfileReport) {
        self.total_time += report.wall_time;
        self.ops.entry(name.to_string()).or_default().record(report);
    }

    pub fn clear(&mut self) {
        self.ops.clear();
        self.total_time = Duration::ZERO;
    }

    pub fn ops(&self) -> &BTreeMap<String, OpStats> {
        &self.ops
    }

    pub fn total_time(&self) -> Duration {
        self.total_time
    }

    pub fn print_summary(&self) {
        println!("\n=== Profiling Summary ===");
        println!(
            "| {:<30} | {:>5} | {:>10} | {:>10} | {:>10} | {:>10} | {:>12} |",
            "Op", "Count", "Avg(µs)", "Min(µs)", "Max(µs)", "Std(µs)", "ScopePeak(B)"
        );
        println!(
            "|{:-<32}|{:-<7}|{:-<12}|{:-<12}|{:-<12}|{:-<12}|{:-<14}|",
            "", "", "", "", "", "", ""
        );

        let mut entries: Vec<_> = self.ops.iter().collect();
        entries.sort_by(|a, b| b.1.total_time_us.cmp(&a.1.total_time_us));

        for (name, stats) in entries {
            println!(
                "| {:<30} | {:>5} | {:>10} | {:>10} | {:>10} | {:>10.1} | {:>12} |",
                truncate_name(name, 30),
                stats.count,
                stats.avg_time_us(),
                stats.min_time_us,
                stats.max_time_us,
                stats.stddev_time_us(),
                stats.max_scope_peak_bytes
            );
        }

        println!(
            "|{:-<32}|{:-<7}|{:-<12}|{:-<12}|{:-<12}|{:-<12}|{:-<14}|",
            "", "", "", "", "", "", ""
        );
        println!("Total profiled time: {:?}\n", self.total_time);
    }
}

fn truncate_name(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        return s;
    }
    let mut end = max_len;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn format_shape(shape: &[usize]) -> String {
    shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("x")
}

fn make_op_key(name: &str, shapes: &[&[usize]]) -> String {
    if shapes.is_empty() {
        return name.to_string();
    }
    let shape_str = shapes
        .iter()
        .map(|s| format_shape(s))
        .collect::<Vec<_>>()
        .join(",");
    format!("{}[{}]", name, shape_str)
}

#[derive(Debug, Clone)]
pub struct ProfiledBackend<B> {
    inner: B,
    registry: Arc<Mutex<Registry>>,
    allocator: Option<&'static TrackingAllocator>,
}

impl<B> ProfiledBackend<B> {
    pub fn new(inner: B, allocator: Option<&'static TrackingAllocator>) -> Self {
        Self {
            inner,
            registry: Arc::new(Mutex::new(Registry::default())),
            allocator,
        }
    }

    pub fn print_report(&self) {
        self.registry.lock().print_summary();
    }

    pub fn clear_stats(&self) {
        self.registry.lock().clear();
    }

    pub fn registry(&self) -> Arc<Mutex<Registry>> {
        Arc::clone(&self.registry)
    }

    fn profile_op<F, R>(&self, name: &str, shapes: &[&[usize]], op: F) -> R
    where
        F: FnOnce() -> R,
    {
        let key = make_op_key(name, shapes);
        let (result, report) = profile(self.allocator, op);
        self.registry.lock().add(&key, &report);
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
        self.profile_op("read", &[layout.shape()], || {
            self.inner.read(storage, layout, dst)
        })
    }

    fn write(&self, storage: &mut Self::Storage, layout: &Layout, src: &[D]) -> Result<()> {
        self.profile_op("write", &[layout.shape()], || {
            self.inner.write(storage, layout, src)
        })
    }
}

impl<D: NativeType, B: CopyOp<D>> CopyOp<D> for ProfiledBackend<B> {
    fn copy(&self, storage: &Self::Storage, layout: &Layout) -> Result<TensorParts<Self::Storage>> {
        self.profile_op("copy", &[layout.shape()], || {
            self.inner.copy(storage, layout)
        })
    }
}

impl<D: NativeType, B: FillOp<D>> FillOp<D> for ProfiledBackend<B> {
    fn fill(&self, layout: &Layout, value: D) -> Result<Self::Storage> {
        self.profile_op("fill", &[layout.shape()], || self.inner.fill(layout, value))
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
        self.profile_op("add", &[lhs_layout.shape(), rhs_layout.shape()], || {
            self.inner.add(lhs, rhs, lhs_layout, rhs_layout)
        })
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
        self.profile_op("sub", &[lhs_layout.shape(), rhs_layout.shape()], || {
            self.inner.sub(lhs, rhs, lhs_layout, rhs_layout)
        })
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
        self.profile_op("matmul", &[lhs_layout.shape(), rhs_layout.shape()], || {
            self.inner.matmul(lhs, rhs, lhs_layout, rhs_layout)
        })
    }
}

impl<D: NativeType, B: MeanOp<D>> MeanOp<D> for ProfiledBackend<B>
where
    B: Backend<f32>,
    ProfiledBackend<B>: Backend<f32>,
{
    fn mean_f32(
        &self,
        storage: &<Self as Backend<D>>::Storage,
        layout: &Layout,
    ) -> Result<TensorParts<<Self as Backend<f32>>::Storage>> {
        let key = make_op_key("mean_f32", &[layout.shape()]);
        let (res, report) = profile(self.allocator, || self.inner.mean_f32(storage, layout));
        self.registry.lock().add(&key, &report);

        let parts = res?;
        let TensorParts {
            storage: inner_storage,
            layout,
        } = parts;

        // SAFETY: ProfiledBackend<B>::Storage is defined as B::Storage in the Backend impl.
        // The transmute copy is a no-op between identical concrete types.
        let outer_storage: <Self as Backend<f32>>::Storage =
            unsafe { std::mem::transmute_copy(&inner_storage) };
        std::mem::forget(inner_storage);

        Ok(TensorParts {
            storage: outer_storage,
            layout,
        })
    }
}
