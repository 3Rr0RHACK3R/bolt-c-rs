use bolt_core::shape::ConcreteShape;
use bolt_core::{DType, Error, Layout, layout::IterMode};

#[test]
fn test_broadcast_to_scalar() {
    let shape = ConcreteShape::from_slice(&[]).unwrap();
    let layout = Layout::contiguous(shape);
    let new_shape = ConcreteShape::from_slice(&[2, 3]).unwrap();
    let new_layout = layout.broadcast_to(&new_shape).unwrap();
    assert_eq!(new_layout.shape(), &[2, 3]);
    assert_eq!(new_layout.strides(), &[0, 0]);
}

#[test]
fn test_broadcast_to_vector() {
    let shape = ConcreteShape::from_slice(&[3]).unwrap();
    let layout = Layout::contiguous(shape);
    let new_shape = ConcreteShape::from_slice(&[2, 3]).unwrap();
    let new_layout = layout.broadcast_to(&new_shape).unwrap();
    assert_eq!(new_layout.shape(), &[2, 3]);
    assert_eq!(new_layout.strides(), &[0, 1]);
}

#[test]
fn test_broadcast_to_matrix() {
    let shape = ConcreteShape::from_slice(&[2, 1]).unwrap();
    let layout = Layout::contiguous(shape);
    let new_shape = ConcreteShape::from_slice(&[2, 3]).unwrap();
    let new_layout = layout.broadcast_to(&new_shape).unwrap();
    assert_eq!(new_layout.shape(), &[2, 3]);
    assert_eq!(new_layout.strides(), &[1, 0]);
}

#[test]
fn test_broadcast_binary() {
    let shape1 = ConcreteShape::from_slice(&[2, 1]).unwrap();
    let layout1 = Layout::contiguous(shape1);
    let shape2 = ConcreteShape::from_slice(&[3]).unwrap();
    let layout2 = Layout::contiguous(shape2);
    let (new_layout1, new_layout2) = Layout::broadcast_binary(&layout1, &layout2).unwrap();
    assert_eq!(new_layout1.shape(), &[2, 3]);
    assert_eq!(new_layout1.strides(), &[1, 0]);
    assert_eq!(new_layout2.shape(), &[2, 3]);
    assert_eq!(new_layout2.strides(), &[0, 1]);
}

#[test]
fn test_iter_offsets_contiguous() {
    let shape = ConcreteShape::from_slice(&[2, 3]).unwrap();
    let layout = Layout::contiguous(shape);
    let offsets: Vec<usize> = layout.iter_offsets(DType::F32).unwrap().collect();
    assert_eq!(offsets, vec![0, 4, 8, 12, 16, 20]);
}

#[test]
fn test_iter_offsets_broadcasted_read() {
    let shape = ConcreteShape::from_slice(&[2, 1]).unwrap();
    let layout = Layout::contiguous(shape);
    let new_shape = ConcreteShape::from_slice(&[2, 3]).unwrap();
    let new_layout = layout.broadcast_to(&new_shape).unwrap();
    let offsets: Vec<usize> = new_layout.iter_offsets(DType::F32).unwrap().collect();
    assert_eq!(offsets, vec![0, 0, 0, 4, 4, 4]);
}

#[test]
fn test_iter_offsets_broadcasted_write() {
    let shape = ConcreteShape::from_slice(&[2, 1]).unwrap();
    let layout = Layout::contiguous(shape);
    let new_shape = ConcreteShape::from_slice(&[2, 3]).unwrap();
    let new_layout = layout.broadcast_to(&new_shape).unwrap();
    assert!(
        new_layout
            .iter_offsets_for(IterMode::Write, DType::F32)
            .is_err()
    );
}

#[test]
fn test_iter_offsets_transpose() {
    let shape = ConcreteShape::from_slice(&[2, 2]).unwrap();
    let layout = Layout::contiguous(shape);
    let new_layout = layout.transpose(0, 1).unwrap();
    let offsets: Vec<usize> = new_layout.iter_offsets(DType::F32).unwrap().collect();
    assert_eq!(offsets, vec![0, 8, 4, 12]);
}

#[test]
fn test_iter_offsets_negative_stride() {
    let shape = ConcreteShape::from_slice(&[5]).unwrap();
    let layout = Layout::with_strides(shape, &[-1], 16).unwrap();
    let offsets: Vec<usize> = layout.iter_offsets(DType::F32).unwrap().collect();
    assert_eq!(offsets, vec![16, 12, 8, 4, 0]);
}

#[test]
fn test_kernel_simulation_usage() {
    let run_kernel = |layout: &Layout| -> Vec<usize> {
        if layout.is_contiguous() {
            let start = layout.offset_bytes();
            let end = start + layout.num_elements() * 4; // F32
            (start..end).step_by(4).collect()
        } else {
            layout.iter_offsets(DType::F32).unwrap().collect()
        }
    };

    let shape = ConcreteShape::from_slice(&[2, 2]).unwrap();
    let layout = Layout::contiguous(shape.clone());
    assert_eq!(run_kernel(&layout), vec![0, 4, 8, 12]);

    let layout_t = layout.transpose(0, 1).unwrap();
    assert!(!layout_t.is_contiguous());
    assert_eq!(run_kernel(&layout_t), vec![0, 8, 4, 12]);
}

#[test]
fn test_iter_offsets_out_of_bounds_negative() {
    let shape = ConcreteShape::from_slice(&[5]).unwrap();
    let layout = Layout::with_strides(shape, &[-1], 0).unwrap();

    let result = layout.iter_offsets(DType::F32);
    assert!(result.is_err());
    match result {
        Err(Error::InvalidShape { message }) => {
            assert!(message.contains("negative memory addresses"));
        }
        _ => panic!("Expected InvalidShape error"),
    }
}

#[test]
fn test_broadcast_to_invalid() {
    let shape = ConcreteShape::from_slice(&[2, 3]).unwrap();
    let layout = Layout::contiguous(shape);
    let new_shape = ConcreteShape::from_slice(&[3]).unwrap();
    assert!(layout.broadcast_to(&new_shape).is_err());
}
