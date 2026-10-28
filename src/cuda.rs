//! Makes the downloaded CUDA runtime (the `cuda-runtime` asset) visible to ONNX Runtime's CUDA provider.
//!
//! The provider finds cuDNN, cuBLAS and the rest by name. Loading each one first, by its full path, puts it in the
//! process under that name, so the provider's own loads resolve to it; no PATH or LD_LIBRARY_PATH, and no CUDA
//! toolkit. A library whose dependencies are not loaded yet fails and is tried again on the next pass.

use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};

static LOADED: OnceLock<Vec<libloading::Library>> = OnceLock::new();

/// Loads every shared library in `dir` (once per process). Call before the first CUDA session.
pub fn use_runtime(dir: &Path) -> Result<()> {
    if LOADED.get().is_some() {
        return Ok(());
    }
    let is_library = |name: &str| name.ends_with(".dll") || name.contains(".so");
    let mut pending: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(is_library))
        .collect();
    pending.sort();
    if pending.is_empty() {
        bail!("no CUDA libraries in {}", dir.display());
    }
    let mut loaded = Vec::new();
    while !pending.is_empty() {
        let before = pending.len();
        let mut failed = Vec::new();
        for path in pending {
            match open(&path) {
                Ok(library) => loaded.push(library),
                Err(e) => failed.push((path, e)),
            }
        }
        if failed.len() == before {
            let (path, e) = &failed[0];
            bail!("cannot load {}: {e}", path.display());
        }
        pending = failed.into_iter().map(|(p, _)| p).collect();
    }
    let _ = LOADED.set(loaded);
    Ok(())
}

#[cfg(windows)]
fn open(path: &Path) -> Result<libloading::Library, libloading::Error> {
    unsafe { libloading::Library::new(path) }
}

#[cfg(unix)]
fn open(path: &Path) -> Result<libloading::Library, libloading::Error> {
    use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};
    // Global, so later loads by soname (the provider's, and these libraries' own) find it.
    unsafe { Library::open(Some(path), RTLD_NOW | RTLD_GLOBAL) }.map(Into::into)
}
