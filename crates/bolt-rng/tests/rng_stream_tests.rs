use bolt_rng::{RngStream, RngStreams};

#[test]
fn rng_stream_same_seed_same_sequence() {
    let mut a = RngStream::from_seed(42);
    let mut b = RngStream::from_seed(42);
    for _ in 0..10_000 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
}

#[test]
fn rng_stream_split_produces_independent_child() {
    let mut parent = RngStream::from_seed(123);
    let mut child = parent.split();
    assert_ne!(parent.next_u64(), child.next_u64());
}

#[test]
fn rng_stream_fold_in_is_stable() {
    let base = RngStream::from_seed(7);
    let mut a = base.fold_in(0);
    let mut b = base.fold_in(0);
    for _ in 0..1024 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
}

#[test]
fn rng_streams_split2_is_stable() {
    let mut s1 = RngStreams::from_seed(999);
    let mut s2 = RngStreams::from_seed(999);
    let (a1, b1) = s1.split2();
    let (a2, b2) = s2.split2();

    let mut a1 = a1;
    let mut b1 = b1;
    let mut a2 = a2;
    let mut b2 = b2;

    for _ in 0..2048 {
        assert_eq!(a1.dropout.next_u64(), a2.dropout.next_u64());
        assert_eq!(b1.dropout.next_u64(), b2.dropout.next_u64());
    }
}
