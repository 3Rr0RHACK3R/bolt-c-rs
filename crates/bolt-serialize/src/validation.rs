use std::collections::HashSet;
use std::path::Path;

use crate::{Error, Result, TensorToSave};

pub fn validate_tensor_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(Error::InvalidName {
            name: name.to_string(),
            reason: "name cannot be empty".to_string(),
        });
    }

    if name.contains('\0') {
        return Err(Error::InvalidName {
            name: name.to_string(),
            reason: "name cannot contain NUL character".to_string(),
        });
    }

    if name.contains('/') || name.contains('\\') {
        return Err(Error::InvalidName {
            name: name.to_string(),
            reason: "name cannot contain path separators (/ or \\)".to_string(),
        });
    }

    if name == ".." || name.contains("..") {
        return Err(Error::InvalidName {
            name: name.to_string(),
            reason: "name cannot contain '..' (parent directory reference)".to_string(),
        });
    }

    if name.starts_with('.') {
        return Err(Error::InvalidName {
            name: name.to_string(),
            reason: "name cannot start with '.'".to_string(),
        });
    }

    Ok(())
}

pub fn validate_no_duplicates<'a>(
    tensors: impl IntoIterator<Item = &'a TensorToSave<'a>>,
) -> Result<()> {
    let mut seen = HashSet::new();
    for t in tensors {
        if !seen.insert(&t.meta.name) {
            return Err(Error::DuplicateName {
                name: t.meta.name.clone(),
            });
        }
    }
    Ok(())
}

pub fn validate_tensor_bytes(tensor: &TensorToSave<'_>) -> Result<()> {
    let expected = tensor.meta.nbytes().ok_or_else(|| Error::NumelOverflow {
        shape: tensor.meta.shape.as_slice().to_vec(),
    })?;
    let actual = tensor.data.len() as u64;

    if expected != actual {
        return Err(Error::ByteSizeMismatch {
            name: tensor.meta.name.clone(),
            expected,
            actual,
            numel: tensor.meta.numel().unwrap_or(0),
            dtype: tensor.meta.dtype,
        });
    }

    Ok(())
}

pub fn validate_shard_path(path: &str, base_dir: &Path) -> Result<()> {
    if path.is_empty() {
        return Err(Error::UnsafePath {
            path: path.to_string(),
            reason: "shard path cannot be empty".to_string(),
        });
    }

    if path.starts_with('/') || path.starts_with('\\') {
        return Err(Error::UnsafePath {
            path: path.to_string(),
            reason: "shard path cannot be absolute".to_string(),
        });
    }

    if path.contains("..") {
        return Err(Error::UnsafePath {
            path: path.to_string(),
            reason: "shard path cannot contain '..' (parent directory escape)".to_string(),
        });
    }

    // Verify the resolved path stays within base_dir
    let full_path = base_dir.join(path);
    let canonical_base = base_dir
        .canonicalize()
        .ok()
        .unwrap_or_else(|| base_dir.to_path_buf());
    let canonical_full = full_path.canonicalize().ok();

    if let Some(ref canonical) = canonical_full
        && !canonical.starts_with(&canonical_base)
    {
        return Err(Error::UnsafePath {
            path: path.to_string(),
            reason: format!(
                "resolved path escapes artifact directory {:?}",
                canonical_base
            ),
        });
    }

    Ok(())
}
