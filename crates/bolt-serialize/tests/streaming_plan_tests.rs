use bolt_core::{DType, shape::Shape};
use bolt_serialize::{Record, RecordMeta, plan_shards};

fn make_record(name: &str, nbytes: usize) -> Record<'static> {
    let numel = nbytes / 4;
    Record::new(
        RecordMeta::new(name, DType::F32, Shape::from_slice(&[numel]).unwrap()),
        vec![0u8; nbytes],
    )
}

#[test]
fn test_plan_shards_single_shard() {
    let records = vec![
        make_record("a", 40),
        make_record("b", 40),
    ];
    
    let shards = plan_shards(&records, Some(1000));
    assert_eq!(shards.len(), 1);
    assert_eq!(shards[0].entries.len(), 2);
    
    assert_eq!(records[shards[0].entries[0].record_idx].meta.name, "a");
    assert_eq!(records[shards[0].entries[1].record_idx].meta.name, "b");
    
    assert_eq!(shards[0].entries[0].start, 0);
    assert_eq!(shards[0].entries[0].end, 40);
    assert_eq!(shards[0].entries[1].start, 40);
    assert_eq!(shards[0].entries[1].end, 80);
}

#[test]
fn test_plan_shards_multiple_shards() {
    let records: Vec<_> = (0..5)
        .map(|i| make_record(&format!("t{}", i), 100))
        .collect();
    
    let shards = plan_shards(&records, Some(250));
    assert!(shards.len() > 1);
    
    for shard in &shards {
        let total: u64 = shard.entries.iter().map(|e| e.end - e.start).sum();
        assert!(total <= 250);
    }
}

#[test]
fn test_plan_shards_sorted_by_name() {
    let records = vec![
        make_record("z_last", 40),
        make_record("a_first", 40),
        make_record("m_middle", 40),
    ];
    
    let shards = plan_shards(&records, Some(1000));
    assert_eq!(shards.len(), 1);
    
    let names: Vec<_> = shards[0].entries.iter()
        .map(|e| records[e.record_idx].meta.name.as_str())
        .collect();
    
    assert_eq!(names, vec!["a_first", "m_middle", "z_last"]);
}
