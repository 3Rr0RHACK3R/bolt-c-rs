use bolt_core::{dtype::DType, error::Result, layout::Layout, shape::ConcreteShape};

#[test]
fn layout_slice_updates_shape_and_offset() -> Result<()> {
    let base = Layout::contiguous(ConcreteShape::from_slice(&[4, 4])?);
    let sliced = base.slice(1, 1, 4, 2, DType::F32)?;
    assert_eq!(sliced.shape(), &[4, 2]);
    assert_eq!(sliced.strides()[1], 2);
    assert_eq!(sliced.offset_bytes(), std::mem::size_of::<f32>());
    Ok(())
}

#[test]
fn layout_permute_swaps_axes() -> Result<()> {
    let base = Layout::contiguous(ConcreteShape::from_slice(&[2, 3, 4])?);
    let permuted = base.permute(&[2, 0, 1])?;
    assert_eq!(permuted.shape(), &[4, 2, 3]);
    assert_eq!(permuted.strides()[0], 1);
    assert_eq!(permuted.strides()[1], 12);
    Ok(())
}

#[test]
fn reshape_requires_contiguous_layout() -> Result<()> {
    let base = Layout::contiguous(ConcreteShape::from_slice(&[2, 6])?);
    let reshaped = base.reshape(ConcreteShape::from_slice(&[3, 4])?)?;
    assert_eq!(reshaped.shape(), &[3, 4]);
    let sliced = reshaped.slice(0, 0, 3, 2, DType::F32)?;
    assert!(sliced.reshape(ConcreteShape::from_slice(&[6])?).is_err());
    Ok(())
}
