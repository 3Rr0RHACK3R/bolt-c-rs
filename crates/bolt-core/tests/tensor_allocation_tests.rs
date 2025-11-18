use bolt_core::{
    device::DeviceKind,
    dtype::DType,
    error::{Error, Result},
    shape::{ConcreteShape, MAX_ELEMENTS, MAX_RANK},
};

mod support;

use support::test_runtime;

#[test]
fn rejects_rank_above_limit() {
    let dims = vec![1usize; MAX_RANK + 1];
    let err = ConcreteShape::from_slice(&dims).expect_err("rank > MAX_RANK must error");
    assert!(matches!(err, Error::InvalidShape { .. }));
}

#[test]
fn accepts_shape_at_element_limit() -> Result<()> {
    let dims = [MAX_ELEMENTS];
    let shape = ConcreteShape::from_slice(&dims)?;
    assert_eq!(shape.num_elements(), MAX_ELEMENTS);
    Ok(())
}

#[test]
fn rejects_shape_exceeding_element_limit() {
    let too_large = MAX_ELEMENTS
        .checked_add(1)
        .expect("MAX_ELEMENTS + 1 must fit in usize");
    let err = ConcreteShape::from_slice(&[too_large]).expect_err("shape should be rejected");
    match err {
        Error::TensorTooLarge { limit, requested } => {
            assert_eq!(limit, MAX_ELEMENTS);
            assert_eq!(requested, too_large);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn runtime_allocation_rejects_dtype_byte_overflow() -> Result<()> {
    let runtime = test_runtime();
    let dtype = DType::F64;
    let elem_limit = usize::MAX / dtype.size_in_bytes();
    let err = match runtime.allocate_uninit(DeviceKind::Cpu, &[elem_limit.saturating_add(1)], dtype)
    {
        Ok(_) => panic!("allocation should fail before device call"),
        Err(err) => err,
    };
    match err {
        Error::TensorTooLarge { limit, requested } => {
            assert_eq!(limit, elem_limit);
            assert_eq!(requested, elem_limit.saturating_add(1));
        }
        other => panic!("unexpected error: {other:?}"),
    }
    Ok(())
}
