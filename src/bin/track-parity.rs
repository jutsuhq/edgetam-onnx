//! Runs an ONNX tracker on the sequences `scripts/parity/reference.py` wrote and compares its masks with
//! transformers' PyTorch ones.
//!
//!     track-parity <reference folder>/<model> --models <asset folder> --tracker edgetam|sam21 [--cpu] [--json out.json]
//!         [--cuda <cuda-runtime asset folder>] [--save <folder for our masks>] [--together <sequence>,<sequence>,…]
//!
//! `--together` follows the named sequences' objects in one EdgeTAM pass over their (shared) frames, each frame encoded
//! once; its ms per frame are for all of them.
//!
//! Done when every frame's mask IoU is at least 0.95, as in edgetam-coreml's parity check.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use edgetam_onnx::frame::iou;
use edgetam_onnx::{Device, Step, VideoTracker, edgetam::EdgeTam, sam21::Sam21};
use serde::{Deserialize, Serialize};

const BOUND: f64 = 0.95;

#[derive(Deserialize)]
struct Sequence {
    sequence: String,
    width: usize,
    height: usize,
    frames: usize,
    points: Vec<(f32, f32, i32)>,
    start: usize,
    gap: serde_json::Value,
}

impl Sequence {
    fn prompt(&self) -> (Vec<[f32; 2]>, Vec<i32>) {
        (self.points.iter().map(|&(x, y, _)| [x, y]).collect(), self.points.iter().map(|p| p.2).collect())
    }
}

#[derive(Serialize)]
struct Report {
    sequence: String,
    frames: usize,
    iou_min: f64,
    iou_mean: f64,
    worst_frame: usize,
    absent_frames: Vec<usize>,
    ms_per_frame: f64,
    seed_ms: f64,
    pass: bool,
    iou_by_frame: Vec<f64>,
}

fn main() -> Result<()> {
    // ONNX Runtime's own log, through ort's tracing: RUST_LOG=ort=debug
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).with_writer(std::io::stderr).init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let root = PathBuf::from(args.first().context("usage: track-parity <reference folder> --models <dir> --tracker <name>")?);
    let models = PathBuf::from(flag("--models").context("--models")?);
    let tracker = flag("--tracker").unwrap_or_else(|| "edgetam".into());
    let device = if args.iter().any(|a| a == "--cpu") { Device::Cpu } else { Device::Cuda };
    if device == Device::Cuda {
        // CUDA 13 and cuDNN 9 libraries (for example from NVIDIA's pip wheels), next to the models by default.
        let runtime = flag("--cuda").map(PathBuf::from).unwrap_or_else(|| models.with_file_name("cuda-runtime"));
        edgetam_onnx::cuda::use_runtime(&runtime)?;
    }

    if let Some(names) = flag("--together") {
        let folders: Vec<PathBuf> = names.split(',').map(|n| root.join(n)).collect();
        let reports = together(&folders, &models, device, flag("--save").map(PathBuf::from).as_deref())?;
        for r in &reports {
            print(r);
        }
        if let Some(path) = flag("--json") {
            std::fs::write(&path, serde_json::to_vec_pretty(&reports)?)?;
        }
        return Ok(());
    }
    let mut folders: Vec<_> = std::fs::read_dir(&root)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.join("sequence.json").is_file()).collect();
    folders.sort();
    if folders.is_empty() {
        bail!("no sequences in {}", root.display());
    }
    let mut reports = Vec::new();
    for folder in folders {
        let seq: Sequence = serde_json::from_slice(&std::fs::read(folder.join("sequence.json"))?)?;
        let mut t: Box<dyn VideoTracker> = match tracker.as_str() {
            "edgetam" => Box::new(EdgeTam::load(&models, device, seq.frames)?),
            "sam21" => Box::new(Sam21::load(&models, device, seq.frames)?),
            other => bail!("unknown tracker {other}"),
        };
        reports.push(run(&folder, &seq, t.as_mut(), flag("--save").map(PathBuf::from).as_deref())?);
        print(reports.last().unwrap());
    }
    if let Some(path) = flag("--json") {
        std::fs::write(&path, serde_json::to_vec_pretty(&reports)?)?;
    }
    if reports.iter().any(|r| !r.pass) {
        std::process::exit(1);
    }
    Ok(())
}

fn print(r: &Report) {
    println!(
        "{:<18} IoU min {:.4} (frame {}) mean {:.4} · {:.1} ms/frame · seed {:.0} ms · absent {:?} · {}",
        r.sequence, r.iou_min, r.worst_frame, r.iou_mean, r.ms_per_frame, r.seed_ms, r.absent_frames,
        if r.pass { "pass" } else { "FAIL" }
    );
}

fn frame(folder: &Path, i: usize) -> Result<image::RgbImage> {
    Ok(image::open(folder.join(format!("frames/{i:05}.jpg")))?.into_rgb8())
}

fn run(folder: &Path, seq: &Sequence, tracker: &mut dyn VideoTracker, save: Option<&Path>) -> Result<Report> {
    let (w, h) = (seq.width, seq.height);
    let (points, labels) = seq.prompt();
    let mut tally = Tally::new(folder, seq, save)?;
    let (mut track_ms, mut seed_ms) = (0.0, 0.0);
    for i in 0..seq.frames {
        let rgb = frame(folder, i)?;
        let start = Instant::now();
        let step = if i == 0 { tracker.seed(rgb.as_raw(), w, h, &points, &labels)? } else { tracker.track(rgb.as_raw(), w, h)? };
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        if i == 0 { seed_ms = ms } else { track_ms += ms }
        tally.add(i, &step)?;
    }
    Ok(tally.report(track_ms, seed_ms))
}

/// Every sequence's object in one pass over the first one's frames.
fn together(folders: &[PathBuf], models: &Path, device: Device, save: Option<&Path>) -> Result<Vec<Report>> {
    let seqs: Vec<Sequence> = folders
        .iter()
        .map(|f| Ok(serde_json::from_slice(&std::fs::read(f.join("sequence.json")).with_context(|| f.display().to_string())?)?))
        .collect::<Result<_>>()?;
    let first = &seqs[0];
    for s in &seqs {
        ensure!(
            (s.width, s.height, s.frames, s.start, &s.gap) == (first.width, first.height, first.frames, first.start, &first.gap),
            "{} has other frames than {}",
            s.sequence,
            first.sequence
        );
    }
    let mut tracker = EdgeTam::load(models, device, first.frames)?;
    let mut tallies: Vec<Tally> = folders.iter().zip(&seqs).map(|(f, s)| Tally::new(f, s, save)).collect::<Result<_>>()?;
    let mut ids = Vec::new();
    let (mut track_ms, mut seed_ms) = (0.0, 0.0);
    for i in 0..first.frames {
        let rgb = frame(&folders[0], i)?;
        let start = Instant::now();
        tracker.encode(rgb.as_raw(), first.width, first.height)?;
        let mut steps = Vec::new();
        for (k, seq) in seqs.iter().enumerate() {
            if i == 0 {
                let (points, labels) = seq.prompt();
                let (id, step) = tracker.add(&points, &labels)?;
                ids.push(id);
                steps.push(step);
            } else {
                steps.push(tracker.step(ids[k])?);
            }
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        if i == 0 { seed_ms = ms } else { track_ms += ms }
        for (tally, step) in tallies.iter_mut().zip(&steps) {
            tally.add(i, step)?;
        }
    }
    Ok(tallies.into_iter().map(|t| t.report(track_ms, seed_ms)).collect())
}

/// One sequence's masks against PyTorch's, frame by frame.
struct Tally<'a> {
    folder: &'a Path,
    seq: &'a Sequence,
    save: Option<PathBuf>,
    ious: Vec<f64>,
    absent: Vec<usize>,
}

impl<'a> Tally<'a> {
    fn new(folder: &'a Path, seq: &'a Sequence, save: Option<&Path>) -> Result<Self> {
        let save = save.map(|d| d.join(&seq.sequence));
        if let Some(dir) = &save {
            std::fs::create_dir_all(dir)?;
        }
        Ok(Self { folder, seq, save, ious: Vec::new(), absent: Vec::new() })
    }

    fn add(&mut self, i: usize, step: &Step) -> Result<()> {
        if step.object_score <= 0.0 {
            self.absent.push(i);
        }
        let theirs: Vec<u8> = image::open(self.folder.join(format!("masks/{i:05}.png")))?.into_luma8().into_raw().into_iter().map(|v| (v > 127) as u8).collect();
        self.ious.push(iou(&step.mask, &theirs));
        if let Some(dir) = &self.save {
            let png: Vec<u8> = step.mask.iter().map(|&v| if v != 0 { 255 } else { 0 }).collect();
            image::GrayImage::from_raw(self.seq.width as u32, self.seq.height as u32, png).unwrap().save(dir.join(format!("{i:05}.png")))?;
        }
        Ok(())
    }

    fn report(self, track_ms: f64, seed_ms: f64) -> Report {
        let ious = self.ious;
        let (worst_frame, iou_min) = ious.iter().copied().enumerate().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
        Report {
            sequence: self.seq.sequence.clone(),
            frames: self.seq.frames,
            iou_min,
            iou_mean: ious.iter().sum::<f64>() / ious.len() as f64,
            worst_frame,
            absent_frames: self.absent,
            // The first tracked frames include CUDA's warm-up; the mean is over every tracked frame all the same.
            ms_per_frame: track_ms / (self.seq.frames - 1).max(1) as f64,
            seed_ms,
            pass: iou_min >= BOUND,
            iou_by_frame: ious.iter().map(|v| (v * 1e4).round() / 1e4).collect(),
        }
    }
}
