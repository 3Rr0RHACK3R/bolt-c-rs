use std::fs::{self, File};
use std::io;
use std::path::Path;

use flate2::read::GzDecoder;
use reqwest::blocking::get;

use crate::{DatasetError, Result};

const BASE_URL: &str = "https://storage.googleapis.com/cvdf-datasets/mnist";

const FILES: [&str; 4] = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
];

fn download_file(dir: &Path, base_name: &str) -> Result<()> {
    let idx_path = dir.join(base_name);
    if idx_path.exists() {
        return Ok(());
    }

    let url = format!("{}/{}.gz", BASE_URL, base_name);
    println!("Downloading {} -> {:?}", url, idx_path);

    let response = get(&url)
        .map_err(|e| DatasetError::Download(format!("request failed: {e}")))?
        .error_for_status()
        .map_err(|e| DatasetError::Download(format!("HTTP error: {e}")))?;

    let bytes = response
        .bytes()
        .map_err(|e| DatasetError::Download(format!("failed to read body: {e}")))?;

    let mut decoder = GzDecoder::new(&bytes[..]);
    let mut out = File::create(&idx_path)?;
    io::copy(&mut decoder, &mut out)?;

    Ok(())
}

pub fn ensure_downloaded(root: impl AsRef<Path>) -> Result<()> {
    let root = root.as_ref();
    fs::create_dir_all(root)?;

    for name in &FILES {
        download_file(root, name)?;
    }

    Ok(())
}
