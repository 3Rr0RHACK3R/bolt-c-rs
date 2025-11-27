use crate::registry::{OpId, OpRecord, Registry};
use parking_lot::Mutex;
use std::sync::Arc;

pub fn print_report(registry: Arc<Mutex<Registry>>) {
    let guard = registry.lock();
    print_time_stats(&guard);
    print_memory_stats(&guard);
}

fn print_time_stats(registry: &Registry) {
    println!("\n=== Time Statistics ===");
    println!(
        "| {:<30} | {:>5} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} |",
        "Op Name", "Count", "H.Avg(us)", "H.Min(us)", "H.Max(us)", "H.Tot(ms)", "D.Avg(us)"
    );
    println!(
        "|{:-<32}|{:-<7}|{:-<12}|{:-<12}|{:-<12}|{:-<12}|{:-<12}|",
        "", "", "", "", "", "", ""
    );

    let mut top_level: Vec<_> = registry.top_level_ops().collect();
    top_level.sort_by(|a, b| b.stats.host_time.total_us.cmp(&a.stats.host_time.total_us));

    for record in top_level {
        print_time_record(registry, record, 0);
        print_time_children(registry, record.id, 1);
    }

    println!(
        "|{:-<32}|{:-<7}|{:-<12}|{:-<12}|{:-<12}|{:-<12}|{:-<12}|",
        "", "", "", "", "", "", ""
    );
    println!("Total Host Time: {:?}", registry.total_time());
}

fn print_time_children(registry: &Registry, parent_id: OpId, depth: usize) {
    let mut children: Vec<_> = registry.children_of(parent_id).collect();
    children.sort_by(|a, b| b.stats.host_time.total_us.cmp(&a.stats.host_time.total_us));

    for child in children {
        print_time_record(registry, child, depth);
        print_time_children(registry, child.id, depth + 1);
    }
}

fn print_time_record(_registry: &Registry, record: &OpRecord, depth: usize) {
    let indent = "  ".repeat(depth);
    let prefix = if depth > 0 { "- " } else { "" };
    let name = format!("{}{}{}", indent, prefix, format_op_name(record));

    let stats = &record.stats;
    let h_time = &stats.host_time;
    let d_time = &stats.device_time;

    let h_total_ms = h_time.total_us as f64 / 1000.0;

    // For device time, if count is 0, print "-"
    let d_avg_str = if d_time.count > 0 {
        format!("{}", d_time.avg_us())
    } else {
        "-".to_string()
    };

    println!(
        "| {:<30} | {:>5} | {:>10} | {:>10} | {:>10} | {:>10.2} | {:>10} |",
        truncate_str(&name, 30),
        h_time.count,
        h_time.avg_us(),
        h_time.min_us,
        h_time.max_us,
        h_total_ms,
        d_avg_str
    );
}

fn print_memory_stats(registry: &Registry) {
    println!("\n=== Memory Statistics ===");
    println!(
        "| {:<30} | {:>8} | {:>6} | {:>8} | {:>6} | {:>8} | {:>8} | {:>8} |",
        "Op Name", "H.Req", "H.Allc", "D.Req", "D.Allc", "H.Peak", "D.Peak", "MaxRSS"
    );
    println!(
        "|{:-<32}|{:-<10}|{:-<8}|{:-<10}|{:-<8}|{:-<10}|{:-<10}|{:-<10}|",
        "", "", "", "", "", "", "", ""
    );

    let mut top_level: Vec<_> = registry.top_level_ops().collect();
    top_level.sort_by(|a, b| b.stats.host_time.total_us.cmp(&a.stats.host_time.total_us));

    for record in top_level {
        print_memory_record(registry, record, 0);
        print_memory_children(registry, record.id, 1);
    }

    println!(
        "|{:-<32}|{:-<10}|{:-<8}|{:-<10}|{:-<8}|{:-<10}|{:-<10}|{:-<10}|",
        "", "", "", "", "", "", "", ""
    );
}

fn print_memory_children(registry: &Registry, parent_id: OpId, depth: usize) {
    let mut children: Vec<_> = registry.children_of(parent_id).collect();
    children.sort_by(|a, b| b.stats.host_time.total_us.cmp(&a.stats.host_time.total_us));

    for child in children {
        print_memory_record(registry, child, depth);
        print_memory_children(registry, child.id, depth + 1);
    }
}

fn print_memory_record(_registry: &Registry, record: &OpRecord, depth: usize) {
    let indent = "  ".repeat(depth);
    let prefix = if depth > 0 { "- " } else { "" };
    let name = format!("{}{}{}", indent, prefix, format_op_name(record));

    let stats = &record.stats;
    let h_mem = &stats.host_memory;
    let d_mem = &stats.device_memory;

    println!(
        "| {:<30} | {:>8} | {:>6} | {:>8} | {:>6} | {:>8} | {:>8} | {:>8} |",
        truncate_str(&name, 30),
        format_bytes(h_mem.total_requested as u64),
        h_mem.alloc_count,
        format_bytes(d_mem.total_requested as u64),
        d_mem.alloc_count,
        format_bytes(h_mem.max_scope_peak),
        format_bytes(d_mem.max_scope_peak),
        format_bytes(stats.max_rss)
    );
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

fn format_bytes(bytes: u64) -> String {
    if bytes == 0 {
        return "0".to_string();
    }
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;

    if bytes >= GB {
        format!("{:.2}G", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2}M", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2}K", bytes as f64 / KB as f64)
    } else {
        format!("{}", bytes)
    }
}
