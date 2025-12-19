use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::{DataError, Result};

pub const MAGIC_IMAGES: u32 = 0x0000_0803;
pub const MAGIC_LABELS: u32 = 0x0000_0801;

pub fn read_u32_be(buf: &[u8]) -> u32 {
    u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]])
}

#[derive(Debug)]
pub struct IdxMeta {
    pub n: usize,
    pub rows: usize,
    pub cols: usize,
}

pub fn read_images_header(path: impl AsRef<Path>) -> Result<(IdxMeta, File)> {
    let mut f = File::open(path)?;

    let mut header = [0u8; 16];
    Read::read_exact(&mut f, &mut header)?;

    let magic = read_u32_be(&header[0..4]);
    if magic != MAGIC_IMAGES {
        return Err(DataError::InvalidMagic {
            expected: MAGIC_IMAGES,
            actual: magic,
        });
    }

    let n = read_u32_be(&header[4..8]) as usize;
    let rows = read_u32_be(&header[8..12]) as usize;
    let cols = read_u32_be(&header[12..16]) as usize;

    Ok((IdxMeta { n, rows, cols }, f))
}

pub fn read_labels_header(path: impl AsRef<Path>) -> Result<(usize, File)> {
    let mut f = File::open(path)?;

    let mut header = [0u8; 8];
    Read::read_exact(&mut f, &mut header)?;

    let magic = read_u32_be(&header[0..4]);
    if magic != MAGIC_LABELS {
        return Err(DataError::InvalidMagic {
            expected: MAGIC_LABELS,
            actual: magic,
        });
    }

    let n = read_u32_be(&header[4..8]) as usize;
    Ok((n, f))
}
