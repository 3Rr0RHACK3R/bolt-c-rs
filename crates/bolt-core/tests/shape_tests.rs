use std::convert::TryFrom;

use bolt_core::{
    error::{Error, Result},
    shape::ConcreteShape,
};

#[test]
fn from_slice_rejects_empty_shapes() {
    let err = ConcreteShape::from_slice(&[]);
    assert!(matches!(err, Err(Error::InvalidShape { .. })));
}

#[test]
fn try_from_vec_rejects_zero_dims() {
    let err = ConcreteShape::try_from(vec![2usize, 0, 3]);
    assert!(matches!(err, Err(Error::InvalidShape { .. })));
}

#[test]
fn try_from_vec_accepts_valid_shapes() -> Result<()> {
    let shape = ConcreteShape::try_from(vec![2usize, 3, 4])?;
    assert_eq!(shape.as_slice(), &[2, 3, 4]);
    Ok(())
}
