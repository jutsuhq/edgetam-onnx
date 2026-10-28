//! EdgeTAM from `jax-image-tools/edgetam-video-onnx`, with every per-frame tensor kept on the GPU.
//!
//! The model's four graphs are chained through IoBinding; the small steps between them (resize and normalise the
//! frame, layouts, the bank's positions, the mask at the video's size) are graphs of our own
//! (`data/glue/*.onnx`, built by `scripts/glue/make_glue.py`). Only the decoded frame goes up, and only the mask
//! and the object score come back.
//!
//! Several objects share one pass of the vision encoder per frame: [`EdgeTam::encode`] runs it once, then
//! [`EdgeTam::add`] seeds a new object on that frame and [`EdgeTam::step`] moves an existing one to it. Each object
//! has only its own bank; every graph and fixed buffer is shared, since objects step one after another and each
//! step's result is in its bank or back on the host before the next one starts.
//!
//! The bank is a ring on the device: spatial slot 0 holds the seed's memory, slots 1–6 the six most recent frames';
//! pointer slot 0 the seed's pointer, the rest the most recent frames'. The memory attention gets a view of the
//! filled slots. Their order does not matter to it (each memory's rotary positions repeat per memory, and pointers
//! carry none); what matters is each memory's temporal row, which `bank_positions` adds per slot from its age. This
//! matches transformers' `EdgeTamVideoModel._prepare_memory_conditioned_features`, which keeps only the frames that
//! exist (no padding).
//!
//! That export's decoder always keeps the best of its three masks. When our own export of it is installed next to the
//! model (`edgetam-decoder/mask_decoder.onnx`, from `scripts/export/edgetam_decoder.py`), it returns all four and the
//! `select_mask` glue picks as transformers does: the best of three for one point or none, the single mask (with its
//! stability fallback) for two or more, as box seeds are.

use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{IoBinding, Session, SessionOutputs};
use ort::value::{DynValue, Shape, Tensor, TensorRefMut, TensorValueType};

use crate::onnx::{self, Device};
use crate::{Step, VideoTracker, frame};

const MEM_DIM: usize = 64;
const BLOCK_TOKENS: usize = 512;
const NUM_MASKMEM: usize = 7;
const MAX_POINTERS: usize = 16;
const POINTER_TOKENS: usize = 4;
/// A full bank: the seed and six recent memories, sixteen pointers.
const FULL_SPATIAL: usize = NUM_MASKMEM * BLOCK_TOKENS;
const FULL_POINTERS: usize = MAX_POINTERS * POINTER_TOKENS;

macro_rules! glue {
    ($name:literal) => {
        include_bytes!(concat!("../data/glue/", $name, ".onnx"))
    };
}

fn err(e: ort::Error) -> anyhow::Error {
    anyhow!("{e}")
}

fn take(out: &mut SessionOutputs<'_>, name: &str) -> Result<DynValue> {
    out.remove(name).with_context(|| format!("no output {name}"))
}

/// A session with its binding.
struct Graph {
    session: Session,
    binding: IoBinding,
}

impl Graph {
    fn new(session: Session) -> Result<Self> {
        let binding = session.create_binding().map_err(err)?;
        Ok(Self { session, binding })
    }

    /// Runs the graph; outputs bound to buffers come back as handles to the same device memory.
    fn run(&mut self) -> Result<SessionOutputs<'_>> {
        let start = std::time::Instant::now();
        let name = std::env::var_os("EDGETAM_ONNX_PROFILE").map(|_| self.session.outputs()[0].name().to_owned());
        let out = self.session.run_binding(&self.binding).map_err(err)?;
        if let Some(name) = name {
            eprintln!("{:>7.2} ms  {name}", start.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(out)
    }
}

/// The encoder's outputs for the current frame: handles to its fixed buffers.
struct Encoded {
    feats0: DynValue,
    feats1: DynValue,
    feats2: DynValue,
    width: usize,
    height: usize,
}

/// One object's memory: its bank on the device and how far it has come.
struct Object {
    spatial: Tensor<f32>,
    pointers: Tensor<f32>,
    /// Frames since the seed; the seed is frame 0.
    frame: usize,
}

/// An object [`EdgeTam::add`] seeded; valid until [`EdgeTam::remove`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectId(usize);

pub struct EdgeTam {
    // Buffers first: they are freed through their allocator, which needs its session alive.
    objects: Vec<Option<Object>>,
    pointer_positions: Tensor<f32>,
    /// The memory attention's graph reads a full bank from here; each object's is copied in.
    stage_spatial: Tensor<f32>,
    stage_pointers: Tensor<f32>,
    _allocator: Allocator,
    /// The banks', from a plain session: once a CUDA graph is recorded, its session's allocator may hand out memory
    /// the graph replays into.
    bank_allocator: Allocator,

    preprocess: Graph,
    encoder: Graph,
    feats_sequence: Graph,
    to_map: Graph,
    add_no_memory: Graph,
    bank_positions: Graph,
    memory_attention: Graph,
    /// Once an object's bank is full its shapes stop changing: the memory attention runs as a CUDA graph on the
    /// stage, with its positions written there too.
    full_positions: Graph,
    full_attention: Graph,
    /// The seed's decoder and memory encoder take any prompt; the tracking ones run on fixed buffers (the memory
    /// encoder as a CUDA graph).
    decoder: Graph,
    memory_encoder: Graph,
    track_decoder: Graph,
    track_memory_encoder: Graph,
    /// With our decoder: the mask choice after the seed's decoder and after the tracking one.
    select: Option<Graph>,
    track_select: Option<Graph>,
    mask_out: Graph,

    device: MemoryInfo<'static>,
    host: MemoryInfo<'static>,
    encoded: Option<Encoded>,
    latent_positions: Option<DynValue>,
    total_frames: usize,
    /// The object the [`VideoTracker`] calls follow.
    single: Option<ObjectId>,
}

impl EdgeTam {
    pub fn load(dir: &Path, device: Device, total_frames: usize) -> Result<Self> {
        let model = |name: &str| onnx::session(&dir.join(format!("{name}.onnx")), device);
        let glue = |bytes: &[u8]| onnx::session_from_memory(bytes, device);
        let mut encoder = Graph::new(onnx::graph_session(&dir.join("vision_encoder.onnx"), device, &[])?)?;
        let device_info = match device {
            Device::Cuda => MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default),
            Device::Cpu => MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::Default),
        }
        .map_err(err)?;
        let host = MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::Default).map_err(err)?;
        let allocator = Allocator::new(&encoder.session, device_info.to_owned()).map_err(err)?;
        let t = |shape: &[usize]| Tensor::<f32>::new(&allocator, shape.to_vec()).map_err(err);

        let all_masks = dir.with_file_name("edgetam-decoder").join("mask_decoder.onnx");
        let all_masks = all_masks.is_file().then_some(all_masks);
        let decoder = || match &all_masks {
            Some(path) => onnx::session(path, device),
            None => model("mask_decoder"),
        };
        // The published decoder has control-flow nodes, which a CUDA graph cannot hold; it still runs on the fixed
        // buffers. Ours has none; tracked frames have no points (one padding point).
        let mut track_decoder = Graph::new(match &all_masks {
            Some(path) => match onnx::graph_session(path, device, &[("num_points", 1)]) {
                Ok(session) => session,
                // An export from before `labels_as_float` keeps two label steps on the CPU.
                Err(_) => onnx::session(path, device)?,
            },
            None => model("mask_decoder")?,
        })?;
        let mut track_memory_encoder = Graph::new(onnx::graph_session(&dir.join("memory_encoder.onnx"), device, &[])?)?;
        let mut feats_sequence = Graph::new(glue(glue!("to_sequence"))?)?;
        let mut add_no_memory = Graph::new(glue(glue!("add_no_memory"))?)?;
        let mut to_map = Graph::new(glue(glue!("to_map"))?)?;
        let mut memory_attention = Graph::new(model("memory_attention")?)?;
        let full = [("spatial_len", FULL_SPATIAL as i64), ("pointer_len", FULL_POINTERS as i64)];
        let mut full_attention = Graph::new(onnx::graph_session(&dir.join("memory_attention.onnx"), device, &full)?)?;
        let mut full_positions = Graph::new(glue(glue!("bank_positions"))?)?;
        let gpu = device_info.allocation_device();

        // Every tensor a frame passes from graph to graph lives in a fixed buffer, bound once: a binding keeps its
        // own reference to an input, so each buffer is bound as an input first and then handed to the binding that
        // writes it. The CUDA graphs replay against these addresses.
        let [feats0, feats1, feats2] = [t(&[1, 32, 256, 256])?, t(&[1, 64, 128, 128])?, t(&[1, 256, 64, 64])?];
        track_decoder.binding.bind_input("feats0", &feats0).map_err(err)?;
        track_decoder.binding.bind_input("feats1", &feats1).map_err(err)?;
        track_memory_encoder.binding.bind_input("vision_features", &feats2).map_err(err)?;
        feats_sequence.binding.bind_input("map", &feats2).map_err(err)?;
        add_no_memory.binding.bind_input("map", &feats2).map_err(err)?;
        // The encoder's position outputs are constants (ORT folds them, and copying a constant out cannot be recorded
        // in a CUDA graph), so they stay unbound here and are taken once from a plain session below.
        encoder.binding.bind_output("feats0", feats0).map_err(err)?;
        encoder.binding.bind_output("feats1", feats1).map_err(err)?;
        encoder.binding.bind_output("feats2", feats2).map_err(err)?;
        // The frame in the memory attention's layout, the same for every object.
        let sequence = t(&[4096, 1, 256])?;
        memory_attention.binding.bind_input("current_vision_features", &sequence).map_err(err)?;
        full_attention.binding.bind_input("current_vision_features", &sequence).map_err(err)?;
        feats_sequence.binding.bind_output("sequence", sequence).map_err(err)?;

        let conditioned_map = t(&[1, 256, 64, 64])?;
        track_decoder.binding.bind_input("feats2", &conditioned_map).map_err(err)?;
        to_map.binding.bind_output("map", conditioned_map).map_err(err)?;
        let no_point = Tensor::from_array((vec![1usize, 1, 1, 2], vec![0f32; 2])).map_err(err)?.to(gpu, 0).map_err(err)?;
        let no_label = Tensor::from_array((vec![1usize, 1, 1], vec![-1i32])).map_err(err)?.to(gpu, 0).map_err(err)?;
        track_decoder.binding.bind_input("input_points", &no_point).map_err(err)?;
        track_decoder.binding.bind_input("input_labels", &no_label).map_err(err)?;
        let track_mask = t(&[1, 1, 1024, 1024])?;
        track_memory_encoder.binding.bind_input("pred_masks_high_res", &track_mask).map_err(err)?;
        let track_select = if all_masks.is_some() {
            let mut select = Graph::new(glue(glue!("select_mask"))?)?;
            let candidates: [(&str, &[usize]); 4] = [
                ("masks", &[1, 1, 4, 256, 256]),
                ("iou_scores", &[1, 1, 4]),
                ("object_pointers", &[1, 1, 4, 256]),
                ("object_score_logits", &[1, 1, 1]),
            ];
            for (name, shape) in candidates {
                let buffer = t(shape)?;
                select.binding.bind_input(name, &buffer).map_err(err)?;
                track_decoder.binding.bind_output(name, buffer).map_err(err)?;
            }
            // Tracked frames have no points: always the best of three.
            let multimask = Tensor::from_array((vec![1usize], vec![true])).map_err(err)?.to(gpu, 0).map_err(err)?;
            select.binding.bind_input("multimask", &multimask).map_err(err)?;
            select.binding.bind_output("high_res_masks", track_mask).map_err(err)?;
            select.binding.bind_output("object_pointer", t(&[1, 1, 256])?).map_err(err)?;
            Some(select)
        } else {
            track_decoder.binding.bind_output("high_res_masks", track_mask).map_err(err)?;
            track_decoder.binding.bind_output("object_pointer", t(&[1, 1, 256])?).map_err(err)?;
            track_decoder.binding.bind_output("object_score_logits", t(&[1, 1, 1])?).map_err(err)?;
            None
        };
        let not_prompted = Tensor::from_array((vec![1usize], vec![false])).map_err(err)?.to(gpu, 0).map_err(err)?;
        track_memory_encoder.binding.bind_input("is_mask_from_pts", &not_prompted).map_err(err)?;
        track_memory_encoder.binding.bind_output("memory_tokens", t(&[1, BLOCK_TOKENS, MEM_DIM])?).map_err(err)?;
        let pos2_sequence = {
            let mut plain = Graph::new(model("vision_encoder")?)?;
            let zeros = Tensor::from_array((vec![1usize, 3, 1024, 1024], vec![0f32; 3 * 1024 * 1024])).map_err(err)?;
            plain.binding.bind_input("pixel_values", &zeros).map_err(err)?;
            plain.binding.bind_output_to_device("pos2", &device_info).map_err(err)?;
            let pos2 = take(&mut plain.run()?, "pos2")?;
            let mut sequence = Graph::new(glue(glue!("to_sequence"))?)?;
            sequence.binding.bind_input("map", &pos2).map_err(err)?;
            sequence.binding.bind_output_to_device("sequence", &device_info).map_err(err)?;
            let out = take(&mut sequence.run()?, "sequence")?;
            // A copy that owns its memory, so the sessions that made it can go.
            out.downcast_ref::<TensorValueType<f32>>().map_err(err)?.to(device_info.allocation_device(), 0).map_err(err)?
        };
        // The image positions in sequence layout: the same on every frame.
        memory_attention.binding.bind_input("current_vision_position_embeddings", &pos2_sequence).map_err(err)?;
        full_attention.binding.bind_input("current_vision_position_embeddings", &pos2_sequence).map_err(err)?;

        let zeros = Tensor::from_array((vec![FULL_POINTERS, 1, MEM_DIM], vec![0f32; FULL_POINTERS * MEM_DIM]))
            .map_err(err)?;
        // The frame's encoder input: written by the preprocessing graph, read by the encoder, always this buffer.
        let pixel = t(&[1, 3, 1024, 1024])?;
        encoder.binding.bind_input("pixel_values", &pixel).map_err(err)?;
        let mut preprocess = Graph::new(glue(glue!("preprocess"))?)?;
        // The encoder's binding holds its own reference to the buffer, so the preprocessing binding can own it.
        preprocess.binding.bind_output("pixel_values", pixel).map_err(err)?;
        let pointer_positions = zeros.to(device_info.allocation_device(), 0).map_err(err)?;

        let [stage_spatial, stage_positions] = [t(&[FULL_SPATIAL, 1, MEM_DIM])?, t(&[FULL_SPATIAL, 1, MEM_DIM])?];
        let stage_pointers = t(&[FULL_POINTERS, 1, MEM_DIM])?;
        let b = &mut full_attention.binding;
        b.bind_input("spatial_memory", &stage_spatial).map_err(err)?;
        b.bind_input("spatial_memory_position_embeddings", &stage_positions).map_err(err)?;
        b.bind_input("pointer_memory", &stage_pointers).map_err(err)?;
        b.bind_input("pointer_memory_position_embeddings", &pointer_positions).map_err(err)?;
        b.bind_output("conditioned_features", t(&[1, 1, 4096, 256])?).map_err(err)?;
        full_positions.binding.bind_output("positions", stage_positions).map_err(err)?;

        Ok(Self {
            objects: Vec::new(),
            pointer_positions,
            stage_spatial,
            stage_pointers,
            _allocator: allocator,
            bank_allocator: Allocator::new(&memory_attention.session, device_info.to_owned()).map_err(err)?,
            preprocess,
            feats_sequence,
            to_map,
            add_no_memory,
            track_decoder,
            track_memory_encoder,
            select: all_masks.is_some().then(|| glue(glue!("select_mask")).and_then(Graph::new)).transpose()?,
            track_select,
            bank_positions: Graph::new(glue(glue!("bank_positions"))?)?,
            mask_out: Graph::new(glue(glue!("mask_out"))?)?,
            memory_attention,
            full_positions,
            full_attention,
            decoder: Graph::new(decoder()?)?,
            memory_encoder: Graph::new(model("memory_encoder")?)?,
            encoder,
            device: device_info,
            host,
            encoded: None,
            latent_positions: None,
            total_frames,
            single: None,
        })
    }

    /// Uploads a frame and encodes it, once for every object on it.
    pub fn encode(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<()> {
        let frame = Tensor::from_array((vec![height, width, 3], rgb.to_vec())).map_err(err)?;
        self.preprocess.binding.bind_input("rgb", &frame).map_err(err)?;
        self.preprocess.run()?;
        let mut out = self.encoder.run()?;
        let encoded = Encoded {
            feats0: take(&mut out, "feats0")?,
            feats1: take(&mut out, "feats1")?,
            feats2: take(&mut out, "feats2")?,
            width,
            height,
        };
        drop(out);
        self.feats_sequence.run()?;
        self.encoded = Some(encoded);
        Ok(())
    }

    /// Seeds a new object on the encoded frame from `points` in the video's pixels (labels as in
    /// [`VideoTracker::seed`]).
    pub fn add(&mut self, points: &[[f32; 2]], labels: &[i32]) -> Result<(ObjectId, Step)> {
        let enc = self.encoded.as_ref().context("add before encode")?;
        let (width, height) = (enc.width, enc.height);
        self.add_no_memory.binding.bind_output_to_device("seed_map", &self.device).map_err(err)?;
        let map = take(&mut self.add_no_memory.run()?, "seed_map")?;
        let points: Vec<[f32; 2]> = points.iter().map(|p| frame::to_model_space(p[0], p[1], width, height)).collect();
        let (mask, score, pointer) = self.decode(&map, &points, labels)?;
        let tokens = self.remember(&mask)?;
        let object = Object {
            spatial: Tensor::new(&self.bank_allocator, vec![FULL_SPATIAL, 1, MEM_DIM]).map_err(err)?,
            pointers: Tensor::new(&self.bank_allocator, vec![FULL_POINTERS, 1, MEM_DIM]).map_err(err)?,
            frame: 0,
        };
        store(&self.device, &object, &tokens, &pointer, 0, 0)?;
        let free = self.objects.iter().position(Option::is_none);
        let index = free.unwrap_or(self.objects.len());
        if free.is_none() {
            self.objects.push(None);
        }
        self.objects[index] = Some(object);
        Ok((ObjectId(index), Step { mask: self.mask_out(&mask)?, object_score: score }))
    }

    /// Moves `id` from its last frame to the encoded one.
    pub fn step(&mut self, id: ObjectId) -> Result<Step> {
        let ring = self.pointer_ring();
        let object = self.objects.get_mut(id.0).and_then(Option::as_mut).context("no such object")?;
        object.frame += 1;
        let t = object.frame;

        // Frames 1…t−1 are in the rings; the seed (frame 0) is never counted twice.
        let recent = (t - 1).min(NUM_MASKMEM - 1);
        let mut rows = vec![(NUM_MASKMEM - 1) as i64; 1 + recent];
        for f in t - recent..t {
            rows[1 + (f - 1) % (NUM_MASKMEM - 1)] = (t - f - 1) as i64;
        }
        let pointer_count = 1 + (t - 1).min(ring);

        let rows = Tensor::from_array((vec![rows.len()], rows)).map_err(err)?;
        let latent_positions = self.latent_positions.as_ref().context("step before add")?;
        let conditioned = if (1 + recent) * BLOCK_TOKENS == FULL_SPATIAL && pointer_count * POINTER_TOKENS == FULL_POINTERS {
            object.spatial.copy_into(&mut self.stage_spatial).map_err(err)?;
            object.pointers.copy_into(&mut self.stage_pointers).map_err(err)?;
            let b = &mut self.full_positions.binding;
            b.bind_input("latent_positions", latent_positions).map_err(err)?;
            b.bind_input("rows", &rows).map_err(err)?;
            self.full_positions.run()?;
            take(&mut self.full_attention.run()?, "conditioned_features")?
        } else {
            let b = &mut self.bank_positions.binding;
            b.bind_input("latent_positions", latent_positions).map_err(err)?;
            b.bind_input("rows", &rows).map_err(err)?;
            b.bind_output_to_device("positions", &self.device).map_err(err)?;
            let positions = take(&mut self.bank_positions.run()?, "positions")?;

            let spatial = raw_view(&object.spatial, &self.device, 0, &[(1 + recent) * BLOCK_TOKENS, 1, MEM_DIM])?;
            let pointers = raw_view(&object.pointers, &self.device, 0, &[pointer_count * POINTER_TOKENS, 1, MEM_DIM])?;
            let pointer_positions =
                raw_view(&self.pointer_positions, &self.device, 0, &[pointer_count * POINTER_TOKENS, 1, MEM_DIM])?;
            let b = &mut self.memory_attention.binding;
            b.bind_input("spatial_memory", &*spatial).map_err(err)?;
            b.bind_input("spatial_memory_position_embeddings", &positions).map_err(err)?;
            b.bind_input("pointer_memory", &*pointers).map_err(err)?;
            b.bind_input("pointer_memory_position_embeddings", &*pointer_positions).map_err(err)?;
            b.bind_output_to_device("conditioned_features", &self.device).map_err(err)?;
            take(&mut self.memory_attention.run()?, "conditioned_features")?
        };

        self.to_map.binding.bind_input("tokens", &conditioned).map_err(err)?;
        self.to_map.run()?;

        let mut out = self.track_decoder.run()?;
        let score = take(&mut out, "object_score_logits")?;
        let (mask, pointer) = match self.track_select.as_mut() {
            None => (take(&mut out, "high_res_masks")?, take(&mut out, "object_pointer")?),
            Some(select) => {
                let mut chosen = select.run()?;
                (take(&mut chosen, "high_res_masks")?, take(&mut chosen, "object_pointer")?)
            }
        };
        drop(out);
        let tokens = take(&mut self.track_memory_encoder.run()?, "memory_tokens")?;
        store(&self.device, object, &tokens, &pointer, 1 + (t - 1) % (NUM_MASKMEM - 1), 1 + (t - 1) % ring)?;
        Ok(Step { mask: self.mask_out(&mask)?, object_score: host_scalar(&score)? })
    }

    /// Frees `id`'s bank.
    pub fn remove(&mut self, id: ObjectId) {
        if let Some(slot) = self.objects.get_mut(id.0) {
            *slot = None;
        }
    }

    /// Decodes `features` with `points` (1024² space): the 1024² logits and the pointer (on the device), and the
    /// object score.
    fn decode(&mut self, features: &DynValue, points: &[[f32; 2]], labels: &[i32]) -> Result<(DynValue, f32, DynValue)> {
        let enc = self.encoded.as_ref().context("decode before encode")?;
        let n = points.len();
        let coords =
            Tensor::from_array((vec![1, 1, n, 2], points.iter().flatten().copied().collect::<Vec<_>>())).map_err(err)?;
        let labels = Tensor::from_array((vec![1, 1, n], labels.to_vec())).map_err(err)?;
        let b = &mut self.decoder.binding;
        b.bind_input("feats0", &enc.feats0).map_err(err)?;
        b.bind_input("feats1", &enc.feats1).map_err(err)?;
        b.bind_input("feats2", features).map_err(err)?;
        b.bind_input("input_points", &coords).map_err(err)?;
        b.bind_input("input_labels", &labels).map_err(err)?;
        let Some(select) = self.select.as_mut() else {
            for name in ["high_res_masks", "object_pointer", "pred_masks", "iou_scores"] {
                b.bind_output_to_device(name, &self.device).map_err(err)?;
            }
            b.bind_output_to_device("object_score_logits", &self.host).map_err(err)?;
            let mut out = self.decoder.run()?;
            let score = out["object_score_logits"].try_extract_tensor::<f32>().map_err(err)?.1[0];
            return Ok((take(&mut out, "high_res_masks")?, score, take(&mut out, "object_pointer")?));
        };
        // transformers' _use_multimask: multimask_min_pt_num 0 <= points <= multimask_max_pt_num 1.
        let multimask = Tensor::from_array((vec![1usize], vec![n <= 1])).map_err(err)?;
        let candidates = ["masks", "iou_scores", "object_pointers", "object_score_logits"];
        for name in candidates {
            b.bind_output_to_device(name, &self.device).map_err(err)?;
        }
        let out = self.decoder.run()?;
        let s = &mut select.binding;
        for name in candidates {
            s.bind_input(name, &out[name]).map_err(err)?;
        }
        s.bind_input("multimask", &multimask).map_err(err)?;
        s.bind_output_to_device("high_res_masks", &self.device).map_err(err)?;
        s.bind_output_to_device("object_pointer", &self.device).map_err(err)?;
        let score = host_scalar(&out["object_score_logits"])?;
        let mut chosen = select.run()?;
        Ok((take(&mut chosen, "high_res_masks")?, score, take(&mut chosen, "object_pointer")?))
    }

    /// The memory tokens of a seed frame's mask.
    fn remember(&mut self, mask: &DynValue) -> Result<DynValue> {
        let enc = self.encoded.as_ref().context("remember before encode")?;
        let flag = Tensor::from_array((vec![1usize], vec![true])).map_err(err)?;
        let b = &mut self.memory_encoder.binding;
        b.bind_input("vision_features", &enc.feats2).map_err(err)?;
        b.bind_input("pred_masks_high_res", mask).map_err(err)?;
        b.bind_input("is_mask_from_pts", &flag).map_err(err)?;
        b.bind_output_to_device("memory_tokens", &self.device).map_err(err)?;
        b.bind_output_to_device("memory_pos_enc", &self.device).map_err(err)?;
        let mut out = self.memory_encoder.run()?;
        let tokens = take(&mut out, "memory_tokens")?;
        if self.latent_positions.is_none() {
            // The latents' positions do not depend on the frame; one copy serves every slot of every object.
            let pos = take(&mut out, "memory_pos_enc")?;
            let pos = pos.downcast_ref::<TensorValueType<f32>>().map_err(err)?;
            self.latent_positions = Some(pos.to(self.device.allocation_device(), 0).map_err(err)?.into_dyn());
        }
        Ok(tokens)
    }

    /// The mask at the video's size, back on the host.
    fn mask_out(&mut self, logits: &DynValue) -> Result<Vec<u8>> {
        let enc = self.encoded.as_ref().context("mask before encode")?;
        let size = Tensor::from_array((vec![2usize], vec![enc.height as i64, enc.width as i64])).map_err(err)?;
        let b = &mut self.mask_out.binding;
        b.bind_input("logits", logits).map_err(err)?;
        b.bind_input("size", &size).map_err(err)?;
        b.bind_output_to_device("mask", &self.host).map_err(err)?;
        let out = self.mask_out.run()?;
        Ok(out["mask"].try_extract_tensor::<u8>().map_err(err)?.1.to_vec())
    }

    /// Recent pointers kept: transformers uses min(frames, 16) − 1 past frames besides the seed.
    fn pointer_ring(&self) -> usize {
        self.total_frames.min(MAX_POINTERS).saturating_sub(1).max(1)
    }
}

impl Drop for EdgeTam {
    /// The buffers bound into the graphs' bindings and the objects' banks came from the allocators, which must outlive
    /// them and must go before the sessions: empty every binding first, then the fields drop in their declared order
    /// (banks and buffers, allocators, sessions).
    fn drop(&mut self) {
        self.encoded = None;
        for graph in [
            &mut self.preprocess,
            &mut self.encoder,
            &mut self.feats_sequence,
            &mut self.to_map,
            &mut self.add_no_memory,
            &mut self.bank_positions,
            &mut self.memory_attention,
            &mut self.full_positions,
            &mut self.full_attention,
            &mut self.decoder,
            &mut self.memory_encoder,
            &mut self.track_decoder,
            &mut self.track_memory_encoder,
            &mut self.mask_out,
        ]
        .into_iter()
        .chain(self.select.as_mut())
        .chain(self.track_select.as_mut())
        {
            graph.binding.clear();
        }
    }
}

/// One object at a time, as the other trackers.
impl VideoTracker for EdgeTam {
    fn seed(&mut self, rgb: &[u8], width: usize, height: usize, points: &[[f32; 2]], labels: &[i32]) -> Result<Step> {
        if let Some(id) = self.single.take() {
            self.remove(id);
        }
        self.encode(rgb, width, height)?;
        let (id, step) = self.add(points, labels)?;
        self.single = Some(id);
        Ok(step)
    }

    fn track(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Step> {
        let id = self.single.context("track before seed")?;
        self.encode(rgb, width, height)?;
        self.step(id)
    }
}

/// Copies a frame's memory and pointer into the object's ring slots, on the device.
fn store(device: &MemoryInfo<'static>, object: &Object, tokens: &DynValue, pointer: &DynValue, slot: usize, pointer_slot: usize) -> Result<()> {
    let mut target = raw_view(&object.spatial, device, slot * BLOCK_TOKENS * MEM_DIM, &[1, BLOCK_TOKENS, MEM_DIM])?;
    tokens.downcast_ref::<TensorValueType<f32>>().map_err(err)?.copy_into(&mut target).map_err(err)?;
    let shape: Vec<usize> = pointer.shape().iter().map(|&d| d as usize).collect();
    let mut target = raw_view(&object.pointers, device, pointer_slot * 256, &shape)?;
    pointer.downcast_ref::<TensorValueType<f32>>().map_err(err)?.copy_into(&mut target).map_err(err)?;
    Ok(())
}

/// The first float of a device tensor, copied to the host.
fn host_scalar(value: &DynValue) -> Result<f32> {
    let value = value.downcast_ref::<TensorValueType<f32>>().map_err(err)?.to(AllocationDevice::CPU, 0).map_err(err)?;
    Ok(value.extract_tensor().1[0])
}

/// A view of `base`'s memory from float `offset`, shaped `shape`. `base` must outlive it.
fn raw_view(base: &Tensor<f32>, info: &MemoryInfo<'static>, offset: usize, shape: &[usize]) -> Result<TensorRefMut<'static, f32>> {
    let full: usize = base.shape().iter().map(|&d| d as usize).product();
    ensure!(offset + shape.iter().product::<usize>() <= full, "view out of range");
    let ptr = unsafe { (base.data_ptr() as *mut f32).add(offset) };
    unsafe { TensorRefMut::from_raw(info.to_owned(), ptr.cast(), Shape::new(shape.iter().map(|&d| d as i64))) }.map_err(err)
}
