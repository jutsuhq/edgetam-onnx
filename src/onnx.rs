//! Thin helpers over ONNX Runtime: sessions on CUDA, float tensors in and out.

use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow};
use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{DynValue, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cuda,
    Cpu,
}

/// The process-wide ONNX Runtime environment, created on first use.
fn environment() -> Result<&'static Environment> {
    static ENV: OnceLock<Environment> = OnceLock::new();
    if let Some(env) = ENV.get() {
        return Ok(env);
    }
    let env = ort::init().with_name("edgetam-onnx").build().map_err(|e| anyhow!("ONNX Runtime: {e}"))?;
    Ok(ENV.get_or_init(|| env))
}

/// Where a graph comes from.
pub enum Source<'a> {
    File(&'a Path),
    /// Our glue graphs, compiled into the binary.
    Memory(&'a [u8]),
}

pub fn session(path: &Path, device: Device) -> Result<Session> {
    build(Source::File(path), device, false, false, &[])
}

/// A session that asks for as little GPU memory as it can: cuDNN's default workspace, and an arena that grows by what
/// is requested. For a large model run once in a while on an 8 GB card: BiRefNet at 1024² filled it and ran 7× slower
/// with the defaults.
pub fn lean_session(path: &Path, device: Device) -> Result<Session> {
    build(Source::File(path), device, false, true, &[])
}

pub fn session_from_memory(bytes: &[u8], device: Device) -> Result<Session> {
    build(Source::Memory(bytes), device, false, false, &[])
}

/// A session whose whole run is recorded once as a CUDA graph and replayed: one launch instead of one per kernel,
/// which keeps the GPU busy when the CPU is. Its inputs and outputs must stay at the same device addresses and
/// shapes for every run. `dims` fixes the graph's named free dimensions, so that ORT can fold the shape arithmetic
/// that would otherwise run on the CPU, which a CUDA graph cannot hold.
pub fn graph_session(path: &Path, device: Device, dims: &[(&str, i64)]) -> Result<Session> {
    build(Source::File(path), device, device == Device::Cuda, false, dims)
}

fn build(source: Source<'_>, device: Device, cuda_graph: bool, lean: bool, dims: &[(&str, i64)]) -> Result<Session> {
    // ort's builder errors carry the builder back and are not Send, so they go through their message.
    let fail = |e: ort::Error<_>| anyhow!("{e}");
    let mut builder = Session::builder(environment()?)
        .map_err(|e| anyhow!("{e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(fail)?;
    for &(name, size) in dims {
        builder = builder.with_dimension_override(name, size).map_err(fail)?;
    }
    if device == Device::Cuda {
        let cuda = ort::ep::CUDA::default()
            .with_tf32(true)
            // Exhaustive search benchmarks kernels on the first run, which a CUDA graph cannot record.
            .with_conv_algorithm_search(if cuda_graph {
                ort::ep::cuda::ConvAlgorithmSearch::Heuristic
            } else {
                ort::ep::cuda::ConvAlgorithmSearch::Exhaustive
            })
            // cuDNN's largest workspace is allocated on demand, which a CUDA graph cannot record either.
            .with_conv_max_workspace(!cuda_graph && !lean)
            .with_arena_extend_strategy(if lean {
                ort::ep::ArenaExtendStrategy::SameAsRequested
            } else {
                ort::ep::ArenaExtendStrategy::NextPowerOfTwo
            })
            .with_cuda_graph(cuda_graph);
        builder = builder.with_execution_providers([cuda.build().error_on_failure()]).map_err(fail)?;
    }
    let name = match source {
        Source::File(path) => path.file_stem().unwrap_or_default().to_string_lossy().into_owned(),
        Source::Memory(_) => "glue".to_owned(),
    };
    // Per-node timings with the provider each node ran on, for finding work that left the GPU.
    if let Some(dir) = std::env::var_os("EDGETAM_ONNX_ORT_PROFILE") {
        builder = builder.with_profiling(Path::new(&dir).join(&name)).map_err(fail)?;
    }
    match source {
        Source::File(path) => {
            builder.commit_from_file(path).map_err(|e| anyhow!("{e}")).with_context(|| format!("loading {}", path.display()))
        }
        Source::Memory(bytes) => builder.commit_from_memory(bytes).map_err(|e| anyhow!("{e}")),
    }
}

/// An owned float tensor: shape and row-major data.
#[derive(Debug, Clone)]
pub struct Array {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Array {
    pub fn new(shape: impl Into<Vec<usize>>, data: Vec<f32>) -> Self {
        let shape = shape.into();
        assert_eq!(shape.iter().product::<usize>(), data.len(), "shape {shape:?}");
        Self { shape, data }
    }

    pub fn zeros(shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        let n = shape.iter().product();
        Self { shape, data: vec![0.0; n] }
    }

    pub fn value(&self) -> Result<DynValue> {
        Ok(Tensor::from_array((self.shape.clone(), self.data.clone().into_boxed_slice()))?.into_dyn())
    }

    /// Same data, another shape.
    pub fn reshaped(mut self, shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        assert_eq!(shape.iter().product::<usize>(), self.data.len(), "reshape to {shape:?}");
        self.shape = shape;
        self
    }
}

pub fn i32_value(shape: Vec<usize>, data: Vec<i32>) -> Result<DynValue> {
    Ok(Tensor::from_array((shape, data.into_boxed_slice()))?.into_dyn())
}

pub fn bool_value(shape: Vec<usize>, data: Vec<bool>) -> Result<DynValue> {
    Ok(Tensor::from_array((shape, data.into_boxed_slice()))?.into_dyn())
}

/// Runs `session` and returns the named outputs as float arrays, in the order asked.
pub fn run(session: &mut Session, inputs: Vec<(&str, DynValue)>, outputs: &[&str]) -> Result<Vec<Array>> {
    let start = std::time::Instant::now();
    let result = session.run(inputs)?;
    if std::env::var_os("EDGETAM_ONNX_PROFILE").is_some() {
        eprintln!("{:>7.1} ms  {}", start.elapsed().as_secs_f64() * 1000.0, outputs.join(","));
    }
    outputs
        .iter()
        .map(|name| {
            let value = result.get(*name).ok_or_else(|| anyhow!("no output {name}"))?;
            let (shape, data) = value.try_extract_tensor::<f32>()?;
            Ok(Array::new(shape.iter().map(|&d| d as usize).collect::<Vec<_>>(), data.to_vec()))
        })
        .collect()
}

/// `[1, C, H, W]` → `[H·W, 1, C]`, the sequence-first layout of the memory attention.
pub fn to_sequence(a: &Array) -> Array {
    let (c, hw) = (a.shape[1], a.shape[2] * a.shape[3]);
    let mut out = vec![0.0; a.data.len()];
    for ch in 0..c {
        for i in 0..hw {
            out[i * c + ch] = a.data[ch * hw + i];
        }
    }
    Array::new([hw, 1, c], out)
}

/// `[N, C]`-ordered tokens (any leading layout with C last) → `[1, C, H, W]`.
pub fn tokens_to_map(data: &[f32], c: usize, h: usize, w: usize) -> Array {
    let hw = h * w;
    assert_eq!(data.len(), hw * c);
    let mut out = vec![0.0; data.len()];
    for i in 0..hw {
        for ch in 0..c {
            out[ch * hw + i] = data[i * c + ch];
        }
    }
    Array::new([1, c, h, w], out)
}
