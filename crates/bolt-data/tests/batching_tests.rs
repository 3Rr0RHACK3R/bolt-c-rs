use bolt_data::{BatchRemainder, DataError, Source, Stream};

struct VecSource<E> {
    items: std::collections::VecDeque<E>,
}

impl<E> VecSource<E> {
    fn new(items: Vec<E>) -> Self {
        Self {
            items: items.into(),
        }
    }
}

impl<E> Source<E> for VecSource<E>
where
    E: Send,
{
    fn next(&mut self) -> Result<Option<E>, DataError> {
        Ok(self.items.pop_front())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.items.len();
        (remaining, Some(remaining))
    }
}

#[test]
fn batch_keeps_remainder() {
    let stream = Stream::new(VecSource::new(vec![1, 2, 3, 4, 5]));
    let batches: Vec<Vec<i32>> = stream
        .batch(2)
        .iter()
        .map(|r| r.expect("batch"))
        .collect();
    assert_eq!(batches, vec![vec![1, 2], vec![3, 4], vec![5]]);
}

#[test]
fn batch_with_drop_last_discards_remainder() {
    let stream = Stream::new(VecSource::new(vec![1, 2, 3, 4, 5]));
    let batches: Vec<Vec<i32>> = stream
        .batch_with(2, BatchRemainder::DropLast)
        .iter()
        .map(|r| r.expect("batch"))
        .collect();
    assert_eq!(batches, vec![vec![1, 2], vec![3, 4]]);
}

#[test]
fn batch_size_zero_errors() {
    let stream = Stream::new(VecSource::new(vec![1, 2, 3]));
    let mut it = stream.batch(0).iter();
    let first = it.next().expect("first item must be error");
    let err = first.expect_err("batch(0) must error");
    match err {
        DataError::InvalidShape(msg) => assert!(msg.contains("batch_size")),
        other => panic!("expected InvalidShape, got {other:?}"),
    }
}

#[test]
fn batch_with_drop_last_size_zero_errors() {
    let stream = Stream::new(VecSource::new(vec![1, 2, 3]));
    let mut it = stream.batch_with(0, BatchRemainder::DropLast).iter();
    let first = it.next().expect("first item must be error");
    let err = first.expect_err("batch_with(0, DropLast) must error");
    match err {
        DataError::InvalidShape(msg) => assert!(msg.contains("batch_size")),
        other => panic!("expected InvalidShape, got {other:?}"),
    }
}
