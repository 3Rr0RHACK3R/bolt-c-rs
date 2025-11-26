use std::cell::RefCell;
use std::sync::Arc;
use std::time::Instant;

use bolt_core::AllocatorDiagnostics;
use bolt_core::allocator::{AllocatorSnapshot, DiagnosticsCaps};
use bolt_core::backend::{AddOp, Backend, CopyOp, FillOp, MatmulOp, MeanOp, SubOp, TensorParts};
use bolt_core::dtype::NativeType;
use bolt_core::error::Result;
use bolt_core::layout::Layout;
use parking_lot::Mutex;

use crate::allocator::{AllocatorStats, TrackingAllocator};
use crate::os_stats::get_os_stats;
use crate::registry::{OpCategory, OpId, Registry, current_scope, pop_scope, push_scope};
use crate::report::{
    DeviceMemoryStats, DeviceTimeStats, HostMemoryStats, HostTimeStats, MemoryStats, ProfileReport,
    TimeStats,
};

thread_local! {
    static SCOPE_START: RefCell<Vec<ScopeState>> = const { RefCell::new(Vec::new()) };
}

struct ScopeState {
    start_time: Instant,
    start_os: crate::os_stats::OsStats,
    start_alloc: Option<AllocatorSnapshot>,
    tracking_start_stats: Option<AllocatorStats>,
    op_id: OpId,
}

pub struct ProfiledBackendBuilder<B> {
    inner: B,
    allocator: Option<&'static TrackingAllocator>,
    sample_rate: f64,
}

impl<B> ProfiledBackendBuilder<B> {
    pub fn new(inner: B) -> Self {
        Self {
            inner,
            allocator: None,
            sample_rate: 1.0,
        }
    }

    pub fn with_tracking_allocator(mut self, allocator: &'static TrackingAllocator) -> Self {
        self.allocator = Some(allocator);
        self
    }

    pub fn sample_rate(mut self, rate: f64) -> Self {
        self.sample_rate = rate.clamp(0.0, 1.0);
        self
    }

    pub fn build(self) -> ProfiledBackend<B> {
        ProfiledBackend {
            inner: self.inner,
            registry: Arc::new(Mutex::new(Registry::default())),
            allocator: self.allocator,
            sample_rate: self.sample_rate,
        }
    }
}

#[derive(Debug)]
pub struct ProfiledBackend<B> {
    inner: B,
    registry: Arc<Mutex<Registry>>,
    allocator: Option<&'static TrackingAllocator>,
    sample_rate: f64,
}

impl<B: Clone> Clone for ProfiledBackend<B> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            registry: Arc::clone(&self.registry),
            allocator: self.allocator,
            sample_rate: self.sample_rate,
        }
    }
}

impl<B> ProfiledBackend<B> {
    pub fn new(inner: B, allocator: Option<&'static TrackingAllocator>) -> Self {
        Self {
            inner,
            registry: Arc::new(Mutex::new(Registry::default())),
            allocator,
            sample_rate: 1.0,
        }
    }

    pub fn builder(inner: B) -> ProfiledBackendBuilder<B> {
        ProfiledBackendBuilder::new(inner)
    }

    pub fn registry(&self) -> Arc<Mutex<Registry>> {
        Arc::clone(&self.registry)
    }

    pub fn clear_stats(&self) {
        self.registry.lock().clear();
    }

    pub fn with_registry<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Registry) -> R,
    {
        let guard = self.registry.lock();
        f(&guard)
    }

    pub fn last_report(&self, id: OpId) -> Option<ProfileReport> {
        self.registry.lock().last_report(id).cloned()
    }
}

impl<B> ProfiledBackend<B> {
    pub fn begin_scope_for<D>(&self, name: &str) -> OpId
    where
        D: NativeType,
        B: Backend<D>,
    {
        let allocator = <B as Backend<D>>::allocator(&self.inner);
        let caps = allocator.capabilities();

        if caps.contains(DiagnosticsCaps::SUPPORTS_SCOPE) {
            allocator.begin_scope();
        }
        let start_alloc = Some(allocator.snapshot());

        let mut tracking_start_stats = None;
        if let Some(ta) = self.allocator {
            ta.begin_scope();
            tracking_start_stats = Some(ta.stats());
        }

        let op_id = self.registry.lock().record(
            name,
            OpCategory::UserScope,
            Vec::new(),
            current_scope(),
            &ProfileReport {
                time: TimeStats {
                    host: HostTimeStats::default(),
                    device: DeviceTimeStats::default(),
                },
                memory: MemoryStats::default(),
            },
        );

        push_scope(op_id);

        SCOPE_START.with(|stack| {
            stack.borrow_mut().push(ScopeState {
                start_time: Instant::now(),
                start_os: get_os_stats(),
                start_alloc,
                tracking_start_stats,
                op_id,
            });
        });

        op_id
    }

    pub fn end_scope_for<D>(&self) -> Option<ProfileReport>
    where
        D: NativeType,
        B: Backend<D>,
    {
        let state = SCOPE_START.with(|stack| stack.borrow_mut().pop())?;
        pop_scope();

        let end_time = Instant::now();
        let end_os = get_os_stats();
        let wall_time = end_time.duration_since(state.start_time);

        let host_time = HostTimeStats {
            wall_time,
            user_time: end_os
                .user_cpu_time
                .saturating_sub(state.start_os.user_cpu_time),
            sys_time: end_os
                .sys_cpu_time
                .saturating_sub(state.start_os.sys_cpu_time),
            thread_time: end_os
                .thread_cpu_time
                .saturating_sub(state.start_os.thread_cpu_time),
            available: true,
        };

        let allocator = <B as Backend<D>>::allocator(&self.inner);
        let caps = allocator.capabilities();

        let device_memory = if caps.contains(DiagnosticsCaps::SUPPORTS_SCOPE) {
            let scope_snapshot = allocator.end_scope();
            let after = allocator.snapshot();
            let base = scope_snapshot
                .unwrap_or_else(|| snapshot_delta(&after, state.start_alloc.as_ref().unwrap()));
            build_device_memory_stats(&base, true)
        } else if let Some(start_snapshot) = state.start_alloc {
            let end_snapshot = allocator.snapshot();
            let base = snapshot_delta(&end_snapshot, &start_snapshot);
            let available = !caps.is_empty() || base.bytes_requested > 0 || base.alloc_count > 0;
            build_device_memory_stats(&base, available)
        } else {
            DeviceMemoryStats::default()
        };

        let host_memory = if let Some(ta) = self.allocator {
            let scope_peak = ta.end_scope();
            let end_stats = ta.stats();
            let start_stats = state.tracking_start_stats.unwrap_or_default();
            build_host_memory_stats(&start_stats, &end_stats, scope_peak)
        } else {
            HostMemoryStats::default()
        };

        let report = ProfileReport {
            time: TimeStats {
                host: host_time,
                device: DeviceTimeStats::default(),
            },
            memory: MemoryStats {
                host: host_memory,
                device: device_memory,
                peak_rss_bytes: end_os.rss_bytes,
            },
        };

        self.registry.lock().update(state.op_id, &report);

        Some(report)
    }
}

impl<B: Backend<f32>> ProfiledBackend<B> {
    pub fn begin_scope(&self, name: &str) -> OpId {
        self.begin_scope_for::<f32>(name)
    }

    pub fn end_scope(&self) -> Option<ProfileReport> {
        self.end_scope_for::<f32>()
    }
}

impl<B> ProfiledBackend<B> {
    fn should_sample(&self) -> bool {
        if self.sample_rate >= 1.0 {
            return true;
        }
        if self.sample_rate <= 0.0 {
            return false;
        }
        rand_simple() < self.sample_rate
    }

    fn begin_profile(&self, allocator: &impl AllocatorDiagnostics) -> Option<ProfileStart> {
        if !self.should_sample() {
            return None;
        }

        let caps = allocator.capabilities();
        if caps.contains(DiagnosticsCaps::SUPPORTS_SCOPE) {
            allocator.begin_scope();
        }
        let alloc = allocator.snapshot();

        let mut tracking_start_stats = None;
        if let Some(ta) = self.allocator {
            ta.begin_scope();
            tracking_start_stats = Some(ta.stats());
        }

        Some(ProfileStart {
            time: Instant::now(),
            os: get_os_stats(),
            alloc,
            tracking_start_stats,
        })
    }

    fn end_profile(
        &self,
        start: Option<ProfileStart>,
        allocator: &impl AllocatorDiagnostics,
        name: &str,
        category: OpCategory,
        shapes: Vec<Vec<usize>>,
        parent: Option<OpId>,
    ) {
        let Some(start) = start else { return };

        let end_time = Instant::now();
        let end_os = get_os_stats();
        let wall_time = end_time.duration_since(start.time);

        let host_time = HostTimeStats {
            wall_time,
            user_time: end_os.user_cpu_time.saturating_sub(start.os.user_cpu_time),
            sys_time: end_os.sys_cpu_time.saturating_sub(start.os.sys_cpu_time),
            thread_time: end_os
                .thread_cpu_time
                .saturating_sub(start.os.thread_cpu_time),
            available: true,
        };

        let caps = allocator.capabilities();
        let device_memory = if caps.contains(DiagnosticsCaps::SUPPORTS_SCOPE) {
            let scope_snapshot = allocator.end_scope();
            let after = allocator.snapshot();
            let base = scope_snapshot.unwrap_or_else(|| snapshot_delta(&after, &start.alloc));
            build_device_memory_stats(&base, true)
        } else {
            let end_snapshot = allocator.snapshot();
            let base = snapshot_delta(&end_snapshot, &start.alloc);
            let available = !caps.is_empty() || base.bytes_requested > 0 || base.alloc_count > 0;
            build_device_memory_stats(&base, available)
        };

        let host_memory = if let Some(ta) = self.allocator {
            let scope_peak = ta.end_scope();
            let end_stats = ta.stats();
            let start_stats = start.tracking_start_stats.unwrap_or_default();
            build_host_memory_stats(&start_stats, &end_stats, scope_peak)
        } else {
            HostMemoryStats::default()
        };

        let report = ProfileReport {
            time: TimeStats {
                host: host_time,
                device: DeviceTimeStats::default(),
            },
            memory: MemoryStats {
                host: host_memory,
                device: device_memory,
                peak_rss_bytes: end_os.rss_bytes,
            },
        };

        self.registry
            .lock()
            .record(name, category, shapes, parent, &report);
    }
}

struct ProfileStart {
    time: Instant,
    os: crate::os_stats::OsStats,
    alloc: AllocatorSnapshot,
    tracking_start_stats: Option<AllocatorStats>,
}

fn rand_simple() -> f64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let state = RandomState::new();
    let mut hasher = state.build_hasher();
    hasher.write_u64(std::time::Instant::now().elapsed().as_nanos() as u64);
    (hasher.finish() as f64) / (u64::MAX as f64)
}

macro_rules! profile_op {
    ($self:expr, $name:literal, $category:expr, $shapes:expr, $op:expr) => {{
        let allocator = $self.inner.allocator();
        let parent = current_scope();
        let start = $self.begin_profile(&allocator);
        let result = $op;
        $self.end_profile(start, &allocator, $name, $category, $shapes, parent);
        result
    }};
}

fn build_device_memory_stats(base: &AllocatorSnapshot, available: bool) -> DeviceMemoryStats {
    DeviceMemoryStats {
        bytes_requested: base.bytes_requested,
        bytes_granted: base.bytes_granted,
        alloc_count: base.alloc_count,
        dealloc_count: base.dealloc_count,
        peak_in_scope: base.peak_in_scope,
        persistent_peak: base.persistent_peak,
        fragmentation_pct: base.fragmentation_pct,
        scratch_bytes: base.scratch_bytes,
        extensions: base.extensions.clone(),
        available,
    }
}

fn build_host_memory_stats(
    start: &crate::allocator::AllocatorStats,
    end: &crate::allocator::AllocatorStats,
    scope_peak: usize,
) -> HostMemoryStats {
    let bytes_requested = end
        .cumulative_allocated_bytes
        .saturating_sub(start.cumulative_allocated_bytes) as u64;
    let alloc_count = end.alloc_count.saturating_sub(start.alloc_count) as u64;
    let dealloc_count = end.dealloc_count.saturating_sub(start.dealloc_count) as u64;

    HostMemoryStats {
        bytes_requested,
        bytes_granted: bytes_requested,
        alloc_count,
        dealloc_count,
        peak_in_scope: scope_peak as u64,
        persistent_peak: end.peak_allocated_bytes as u64,
        available: true,
    }
}

fn snapshot_delta(after: &AllocatorSnapshot, before: &AllocatorSnapshot) -> AllocatorSnapshot {
    AllocatorSnapshot {
        bytes_requested: after.bytes_requested.saturating_sub(before.bytes_requested),
        bytes_granted: after.bytes_granted.saturating_sub(before.bytes_granted),
        alloc_count: after.alloc_count.saturating_sub(before.alloc_count),
        dealloc_count: after.dealloc_count.saturating_sub(before.dealloc_count),
        peak_in_scope: after.peak_in_scope.saturating_sub(before.peak_in_scope),
        persistent_peak: after.persistent_peak,
        fragmentation_pct: after.fragmentation_pct,
        scratch_bytes: after.scratch_bytes,
        extensions: after.extensions.clone(),
    }
}

fn shapes_from_layout(layout: &Layout) -> Vec<usize> {
    layout.shape().to_vec()
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
        profile_op!(
            self,
            "read",
            OpCategory::Memory,
            vec![shapes_from_layout(layout)],
            self.inner.read(storage, layout, dst)
        )
    }

    fn write(&self, storage: &mut Self::Storage, layout: &Layout, src: &[D]) -> Result<()> {
        profile_op!(
            self,
            "write",
            OpCategory::Memory,
            vec![shapes_from_layout(layout)],
            self.inner.write(storage, layout, src)
        )
    }
}

impl<D: NativeType, B: CopyOp<D>> CopyOp<D> for ProfiledBackend<B> {
    fn copy(&self, storage: &Self::Storage, layout: &Layout) -> Result<TensorParts<Self::Storage>> {
        profile_op!(
            self,
            "copy",
            OpCategory::Memory,
            vec![shapes_from_layout(layout)],
            self.inner.copy(storage, layout)
        )
    }
}

impl<D: NativeType, B: FillOp<D>> FillOp<D> for ProfiledBackend<B> {
    fn fill(&self, layout: &Layout, value: D) -> Result<Self::Storage> {
        profile_op!(
            self,
            "fill",
            OpCategory::Memory,
            vec![shapes_from_layout(layout)],
            self.inner.fill(layout, value)
        )
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
        profile_op!(
            self,
            "add",
            OpCategory::Compute,
            vec![
                shapes_from_layout(lhs_layout),
                shapes_from_layout(rhs_layout)
            ],
            self.inner.add(lhs, rhs, lhs_layout, rhs_layout)
        )
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
        profile_op!(
            self,
            "sub",
            OpCategory::Compute,
            vec![
                shapes_from_layout(lhs_layout),
                shapes_from_layout(rhs_layout)
            ],
            self.inner.sub(lhs, rhs, lhs_layout, rhs_layout)
        )
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
        profile_op!(
            self,
            "matmul",
            OpCategory::Compute,
            vec![
                shapes_from_layout(lhs_layout),
                shapes_from_layout(rhs_layout)
            ],
            self.inner.matmul(lhs, rhs, lhs_layout, rhs_layout)
        )
    }
}

impl<D: NativeType, B: MeanOp<D>> MeanOp<D> for ProfiledBackend<B> {
    type F32Storage = B::F32Storage;

    fn mean_f32(
        &self,
        storage: &<Self as Backend<D>>::Storage,
        layout: &Layout,
    ) -> Result<TensorParts<Self::F32Storage>> {
        let allocator = self.inner.allocator();
        let parent = current_scope();
        let start = self.begin_profile(&allocator);

        let result = self.inner.mean_f32(storage, layout);

        self.end_profile(
            start,
            &allocator,
            "mean_f32",
            OpCategory::Compute,
            vec![shapes_from_layout(layout)],
            parent,
        );

        result
    }
}
