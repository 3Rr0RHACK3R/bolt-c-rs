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
fn layout_slice_overflow_positive_stride_still_ok() -> Result<()> {
    let huge_stride = isize::MAX / 4;
    let start = (isize::MAX / 2) as usize;
    let shape = ConcreteShape::from_slice(&[start + 1, 1])?;
    let layout = Layout::with_strides(shape, &[huge_stride, 1], 0)?;

    let result = layout.slice(0, start, start + 1, 1, DType::F64);
    assert!(result.is_err(), "slice unexpectedly succeeded despite overflowing stride math");
    Ok(())
}

#[test]
fn layout_slice_reports_stride_mul_overflow_error() -> Result<()> {
    let (layout, start) = huge_stride_layout(isize::MAX / 4)?;
    let err = layout
        .slice(0, start, start + 1, 1, DType::F64)
        .expect_err("expected stride/start overflow");
    assert!(matches!(
        err,
        bolt_core::error::Error::InvalidShape { ref message }
            if message == "slice offset overflow (stride*start)"
    ));
    Ok(())
}

#[test]
fn layout_slice_reports_byte_mul_overflow_error() -> Result<()> {
    let stride = (isize::MAX / 8) - 1;
    let start = 8usize;
    let shape = ConcreteShape::from_slice(&[start + 1, 1])?;
    let layout = Layout::with_strides(shape, &[stride, 1], 0)?;
    let err = layout
        .slice(0, start, start + 1, 1, DType::F64)
        .expect_err("expected byte overflow");
    assert!(matches!(
        err,
        bolt_core::error::Error::InvalidShape { ref message }
            if message == "slice offset overflow (bytes)"
    ));
    Ok(())
}

#[test]
fn layout_slice_reports_base_plus_delta_overflow_error() -> Result<()> {
    let dtype = DType::I32;
    let shape = ConcreteShape::from_slice(&[4])?;
    let layout = Layout::with_strides(shape, &[1], usize::MAX - 2)?;
    let err = layout
        .slice(0, 1, 2, 1, dtype)
        .expect_err("expected base+delta overflow");
    assert!(matches!(
        err,
        bolt_core::error::Error::InvalidShape { ref message }
            if message == "slice offset overflow (base+delta)"
    ));
    Ok(())
}

#[test]
fn layout_slice_supports_negative_stride_offsets() -> Result<()> {
    let dtype = DType::F32;
    let shape = ConcreteShape::from_slice(&[8])?;
    let base_offset = (shape.as_slice()[0] - 1) * dtype.size_in_bytes();
    let layout = Layout::with_strides(shape, &[-1], base_offset)?;
    let sliced = layout.slice(0, 2, 4, 1, dtype)?;
    assert_eq!(sliced.strides()[0], -1);
    assert_eq!(
        sliced.offset_bytes(),
        base_offset - 2 * dtype.size_in_bytes()
    );
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

#[test]
fn reshape_rejects_invalid_element_counts() -> Result<()> {
    let base = Layout::contiguous(ConcreteShape::from_slice(&[2, 3])?);
    let err = base.reshape(ConcreteShape::from_slice(&[2, 2])?);
    assert!(matches!(
        err,
        Err(bolt_core::error::Error::SizeMismatch { .. })
    ));
    Ok(())
}

fn huge_stride_layout(stride: isize) -> Result<(Layout, usize)> {
    let start = (isize::MAX / 2) as usize;
    let shape = ConcreteShape::from_slice(&[start + 1, 1])?;
    let layout = Layout::with_strides(shape, &[stride, 1], 0)?;
    Ok((layout, start))
}
