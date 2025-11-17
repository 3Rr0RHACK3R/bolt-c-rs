use std::{sync::Arc, thread};

use bolt_core::device::Device;
use bolt_cpu::CpuDevice;

#[test]
fn concurrent_bidirectional_copy() {
    let device = Arc::new(CpuDevice::new());
    let size = 1024 * 1024;

    let buf_a = device.alloc(size, 1).unwrap();
    let buf_b = device.alloc(size, 1).unwrap();

    device.write(buf_a, 0, &vec![1u8; size]).unwrap();
    device.write(buf_b, 0, &vec![2u8; size]).unwrap();

    let device_a_to_b: Arc<CpuDevice> = Arc::clone(&device);
    let device_b_to_a: Arc<CpuDevice> = Arc::clone(&device);

    let handle_ab = thread::spawn(move || {
        for _ in 0..100 {
            device_a_to_b
                .copy(buf_a, 0, buf_b, 0, size)
                .expect("copy A->B succeeds");
        }
    });

    let handle_ba = thread::spawn(move || {
        for _ in 0..100 {
            device_b_to_a
                .copy(buf_b, 0, buf_a, 0, size)
                .expect("copy B->A succeeds");
        }
    });

    handle_ab.join().unwrap();
    handle_ba.join().unwrap();

    let mut data_a = vec![0u8; size];
    let mut data_b = vec![0u8; size];
    device.read(buf_a, 0, &mut data_a).unwrap();
    device.read(buf_b, 0, &mut data_b).unwrap();

    let ones = vec![1u8; size];
    let twos = vec![2u8; size];
    assert!(data_a == ones || data_a == twos);
    assert!(data_b == ones || data_b == twos);
}

#[test]
fn same_buffer_overlap_copy_matches_copy_within() {
    let device = CpuDevice::new();
    let size = 32;
    let buf = device.alloc(size, 1).unwrap();

    let pattern: Vec<u8> = (0..size as u8).collect();
    device.write(buf, 0, &pattern).unwrap();

    device.copy(buf, 0, buf, 8, 16).unwrap();

    let mut actual = vec![0u8; size];
    device.read(buf, 0, &mut actual).unwrap();

    let mut expected = pattern.clone();
    expected.copy_within(0..16, 8);

    assert_eq!(actual, expected);
}
