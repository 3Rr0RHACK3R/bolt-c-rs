use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::Duration;

use crate::report::{MemoryStatsSource, ProfileReport};

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
pub struct OpStats {
    pub count: usize,
    pub total_time_us: u128,
    pub min_time_us: u128,
    pub max_time_us: u128,
    pub sum_sq_time_us: u128,
    pub total_requested_bytes: u128,
    pub total_granted_bytes: u128,
    pub max_scope_peak_bytes: u64,
    pub last_report: Option<ProfileReport>,
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
        self.total_requested_bytes += report.memory_stats.bytes_requested as u128;
        self.total_granted_bytes += report.memory_stats.bytes_granted as u128;
        self.max_scope_peak_bytes = self
            .max_scope_peak_bytes
            .max(report.memory_stats.peak_in_scope);
        self.last_report = Some(report.clone());
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

    pub fn avg_granted_bytes(&self) -> u64 {
        if self.count == 0 {
            0
        } else {
            (self.total_granted_bytes / self.count as u128) as u64
        }
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
        self.total_time += report.wall_time;

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
            self.total_time += report.wall_time;
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
        self.next_id = 0;
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

    pub fn print_summary(&self) {
        self.print_summary_impl(false);
    }

    pub fn print_summary_detailed(&self) {
        self.print_summary_impl(true);
    }

    fn print_summary_impl(&self, detailed: bool) {
        println!("\n=== Profiling Summary ===");
        println!(
            "| {:<30} | {:>5} | {:>10} | {:>10} | {:>10} | {:>10} | {:>12} | {:>6} |",
            "Op", "Count", "Avg(us)", "Min(us)", "Max(us)", "Std(us)", "ScopePeak(B)", "Source"
        );
        println!(
            "|{:-<32}|{:-<7}|{:-<12}|{:-<12}|{:-<12}|{:-<12}|{:-<14}|{:-<8}|",
            "", "", "", "", "", "", "", ""
        );

        let mut top_level: Vec<_> = self.top_level_ops().collect();
        top_level.sort_by(|a, b| b.stats.total_time_us.cmp(&a.stats.total_time_us));

        for record in top_level {
            self.print_record(record, 0);
            if detailed {
                self.print_children(record.id, 1);
            }
        }

        println!(
            "|{:-<32}|{:-<7}|{:-<12}|{:-<12}|{:-<12}|{:-<12}|{:-<14}|{:-<8}|",
            "", "", "", "", "", "", "", ""
        );
        println!("Total profiled time: {:?}\n", self.total_time);
    }

    fn print_children(&self, parent: OpId, depth: usize) {
        let mut children: Vec<_> = self.children_of(parent).collect();
        children.sort_by(|a, b| b.stats.total_time_us.cmp(&a.stats.total_time_us));
        for child in children {
            self.print_record(child, depth);
            self.print_children(child.id, depth + 1);
        }
    }

    fn print_record(&self, record: &OpRecord, depth: usize) {
        let indent = "  ".repeat(depth);
        let prefix = if depth > 0 { "- " } else { "" };
        let name = format!("{}{}{}", indent, prefix, format_op_name(record));
        let src = record
            .stats
            .last_report
            .as_ref()
            .map(|r| format_source(r.memory_stats.source))
            .unwrap_or("-");
        println!(
            "| {:<30} | {:>5} | {:>10} | {:>10} | {:>10} | {:>10.1} | {:>12} | {:>6} |",
            truncate_str(&name, 30),
            record.stats.count,
            record.stats.avg_time_us(),
            record.stats.min_time_us,
            record.stats.max_time_us,
            record.stats.stddev_time_us(),
            record.stats.max_scope_peak_bytes,
            src
        );
    }

    pub fn print_memory_details(&self) {
        println!("Memory stats legend:");
        println!("  - source: backend | tracking | none");
        println!("  - bytes_requested/granted: allocator-reported totals");
        println!("  - alloc/dealloc_count: operation-level allocation counts");
        println!("  - peak_in_scope: scoped high-water mark during the op");
        println!("  - persistent_peak: allocator lifetime high-water mark\n");

        println!(
            "| {:<30} | {:>6} | {:>12} | {:>12} | {:>6} | {:>6} | {:>12} | {:>12} |",
            "Op", "Source", "Req(B)", "Grant(B)", "Alloc", "Dealloc", "PeakScope", "Persistent"
        );
        println!(
            "|{:-<32}|{:-<8}|{:-<14}|{:-<14}|{:-<8}|{:-<8}|{:-<14}|{:-<14}|",
            "", "", "", "", "", "", "", ""
        );

        let mut top_level: Vec<_> = self.top_level_ops().collect();
        top_level.sort_by(|a, b| b.stats.total_time_us.cmp(&a.stats.total_time_us));

        for record in top_level {
            self.print_memory_record(record, 0);
        }

        println!(
            "|{:-<32}|{:-<8}|{:-<14}|{:-<14}|{:-<8}|{:-<8}|{:-<14}|{:-<14}|",
            "", "", "", "", "", "", "", ""
        );
    }

    fn print_memory_record(&self, record: &OpRecord, depth: usize) {
        if let Some(report) = &record.stats.last_report {
            let indent = "  ".repeat(depth);
            let prefix = if depth > 0 { "- " } else { "" };
            let name = format!("{}{}{}", indent, prefix, format_op_name(record));
            let mem = &report.memory_stats;
            println!(
                "| {:<30} | {:>6} | {:>12} | {:>12} | {:>6} | {:>6} | {:>12} | {:>12} |",
                truncate_str(&name, 30),
                format_source(mem.source),
                mem.bytes_requested,
                mem.bytes_granted,
                mem.alloc_count,
                mem.dealloc_count,
                mem.peak_in_scope,
                mem.persistent_peak,
            );
        }
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
                    if r.stats.avg_time_us() < min {
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

fn format_op_name(record: &OpRecord) -> String {
    if record.shapes.is_empty() {
        record.name.clone()
    } else {
        let shapes_str = record
            .shapes
            .iter()
            .map(|s| {
                s.iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join("x")
            })
            .collect::<Vec<_>>()
            .join(",");
        format!("{}[{}]", record.name, shapes_str)
    }
}

fn truncate_str(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        return s;
    }
    let mut end = max_len;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn format_source(source: MemoryStatsSource) -> &'static str {
    match source {
        MemoryStatsSource::BackendAllocator => "backend",
        MemoryStatsSource::TrackingAllocator => "tracking",
        MemoryStatsSource::Unavailable => "none",
    }
}
