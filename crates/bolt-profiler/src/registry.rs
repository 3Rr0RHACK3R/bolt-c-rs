use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::Duration;

use crate::report::ProfileReport;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OpId(pub(crate) u64);

impl OpId {
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpCategory {
    #[default]
    Compute,
    Memory,
    Sync,
    UserScope,
}

#[derive(Debug, Clone)]
pub struct OpRecord {
    pub id: OpId,
    pub name: String,
    pub category: OpCategory,
    pub shapes: Vec<Vec<usize>>,
    pub parent: Option<OpId>,
    pub stats: OpStats,
}

#[derive(Debug, Clone, Default)]
pub struct TimeAgg {
    pub total_us: u128,
    pub min_us: u128,
    pub max_us: u128,
    pub sum_sq_us: u128,
    pub count: usize,
}

impl TimeAgg {
    pub fn record(&mut self, us: u128) {
        self.count += 1;
        self.total_us += us;
        if self.count == 1 {
            self.min_us = us;
        } else {
            self.min_us = self.min_us.min(us);
        }
        self.max_us = self.max_us.max(us);
        self.sum_sq_us += us * us;
    }

    pub fn avg_us(&self) -> u128 {
        if self.count == 0 {
            0
        } else {
            self.total_us / self.count as u128
        }
    }

    pub fn stddev_us(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let n = self.count as f64;
        let mean = self.total_us as f64 / n;
        let variance = (self.sum_sq_us as f64 / n) - (mean * mean);
        variance.max(0.0).sqrt()
    }
}

#[derive(Debug, Clone, Default)]
pub struct MemoryAgg {
    pub total_requested: u128,
    pub total_granted: u128,
    pub alloc_count: usize,
    pub dealloc_count: usize,
    pub max_scope_peak: u64,
    pub max_persistent_peak: u64,
}

impl MemoryAgg {
    pub fn record(
        &mut self,
        req: u64,
        grant: u64,
        alloc: u64,
        dealloc: u64,
        peak: u64,
        persist: u64,
    ) {
        self.total_requested += req as u128;
        self.total_granted += grant as u128;
        self.alloc_count += alloc as usize;
        self.dealloc_count += dealloc as usize;
        self.max_scope_peak = self.max_scope_peak.max(peak);
        self.max_persistent_peak = self.max_persistent_peak.max(persist);
    }
}

#[derive(Debug, Clone, Default)]
pub struct OpStats {
    pub count: usize,
    pub host_time: TimeAgg,
    pub device_time: TimeAgg,
    pub host_memory: MemoryAgg,
    pub device_memory: MemoryAgg,
    pub max_rss: u64,
    pub last_report: Option<ProfileReport>,
}

impl OpStats {
    pub fn record(&mut self, report: &ProfileReport) {
        self.count += 1;

        if report.time.host.available {
            self.host_time
                .record(report.time.host.wall_time.as_micros());
        }

        if report.time.device.available {
            // Assuming kernel time as main metric if available, or 0
            let t = report
                .time
                .device
                .kernel_time
                .map(|d| d.as_micros())
                .unwrap_or(0);
            self.device_time.record(t);
        }

        if report.memory.host.available {
            self.host_memory.record(
                report.memory.host.bytes_requested,
                report.memory.host.bytes_granted,
                report.memory.host.alloc_count,
                report.memory.host.dealloc_count,
                report.memory.host.peak_in_scope,
                report.memory.host.persistent_peak,
            );
        }

        if report.memory.device.available {
            self.device_memory.record(
                report.memory.device.bytes_requested,
                report.memory.device.bytes_granted,
                report.memory.device.alloc_count,
                report.memory.device.dealloc_count,
                report.memory.device.peak_in_scope,
                report.memory.device.persistent_peak,
            );
        }

        self.max_rss = self.max_rss.max(report.memory.peak_rss_bytes);
        self.last_report = Some(report.clone());
    }

    pub fn avg_host_time_us(&self) -> u128 {
        self.host_time.avg_us()
    }
}

thread_local! {
    static SCOPE_STACK: RefCell<Vec<OpId>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn current_scope() -> Option<OpId> {
    SCOPE_STACK.with(|s| s.borrow().last().copied())
}

pub(crate) fn push_scope(id: OpId) {
    SCOPE_STACK.with(|s| s.borrow_mut().push(id));
}

pub(crate) fn pop_scope() -> Option<OpId> {
    SCOPE_STACK.with(|s| s.borrow_mut().pop())
}

#[derive(Default, Debug)]
pub struct Registry {
    ops: BTreeMap<OpId, OpRecord>,
    next_id: u64,
    total_time: Duration,
}

impl Registry {
    pub fn record(
        &mut self,
        name: &str,
        category: OpCategory,
        shapes: Vec<Vec<usize>>,
        parent: Option<OpId>,
        report: &ProfileReport,
    ) -> OpId {
        let id = OpId(self.next_id);
        self.next_id += 1;
        self.total_time += report.time.host.wall_time;

        let mut stats = OpStats::default();
        stats.record(report);

        self.ops.insert(
            id,
            OpRecord {
                id,
                name: name.to_string(),
                category,
                shapes,
                parent,
                stats,
            },
        );
        id
    }

    pub fn update(&mut self, id: OpId, report: &ProfileReport) {
        if let Some(record) = self.ops.get_mut(&id) {
            self.total_time += report.time.host.wall_time;
            record.stats.record(report);
        }
    }

    pub fn get(&self, id: OpId) -> Option<&OpRecord> {
        self.ops.get(&id)
    }

    pub fn last_report(&self, id: OpId) -> Option<&ProfileReport> {
        self.ops.get(&id).and_then(|r| r.stats.last_report.as_ref())
    }

    pub fn clear(&mut self) {
        self.ops.clear();
        // Preserve next_id to avoid colliding with stale OpId references
        self.total_time = Duration::ZERO;
    }

    pub fn ops(&self) -> &BTreeMap<OpId, OpRecord> {
        &self.ops
    }

    pub fn total_time(&self) -> Duration {
        self.total_time
    }

    pub fn top_level_ops(&self) -> impl Iterator<Item = &OpRecord> {
        self.ops.values().filter(|r| r.parent.is_none())
    }

    pub fn children_of(&self, parent: OpId) -> impl Iterator<Item = &OpRecord> {
        self.ops.values().filter(move |r| r.parent == Some(parent))
    }
}

pub struct QueryBuilder<'a> {
    registry: &'a Registry,
    category: Option<OpCategory>,
    min_duration_us: Option<u128>,
    top_level_only: bool,
    name_contains: Option<String>,
}

impl<'a> QueryBuilder<'a> {
    pub fn new(registry: &'a Registry) -> Self {
        Self {
            registry,
            category: None,
            min_duration_us: None,
            top_level_only: false,
            name_contains: None,
        }
    }

    pub fn category(mut self, cat: OpCategory) -> Self {
        self.category = Some(cat);
        self
    }

    pub fn min_duration(mut self, duration: Duration) -> Self {
        self.min_duration_us = Some(duration.as_micros());
        self
    }

    pub fn top_level_only(mut self) -> Self {
        self.top_level_only = true;
        self
    }

    pub fn name_contains(mut self, substr: &str) -> Self {
        self.name_contains = Some(substr.to_string());
        self
    }

    pub fn collect(self) -> Vec<&'a OpRecord> {
        self.registry
            .ops
            .values()
            .filter(|r| {
                if self.top_level_only && r.parent.is_some() {
                    return false;
                }
                if let Some(cat) = self.category {
                    if r.category != cat {
                        return false;
                    }
                }
                if let Some(min) = self.min_duration_us {
                    if r.stats.avg_host_time_us() < min {
                        return false;
                    }
                }
                if let Some(ref substr) = self.name_contains {
                    if !r.name.contains(substr) {
                        return false;
                    }
                }
                true
                })
                .collect()
                }
                }
