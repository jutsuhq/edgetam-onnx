//! SAM 2.1 tiny from `square-zero-labs/sam2.1-tiny-video-onnx`, driven the way its README describes. Its memory
//! attention has fixed shapes, so the bank is always 7 memories and 16 pointers, padded by repeating the most recent.

use std::collections::VecDeque;
use std::path::Path;

use anyhow::{Context, Result};
use ort::session::Session;

use crate::onnx::{self, Array, Device};
use crate::{Memory, Step, VideoTracker, frame};

const TOKENS: usize = 64 * 64;
const MEM_DIM: usize = 64;
const BLOCKS: usize = 7;
const POINTERS: usize = 16;
const SPLITS: usize = 256 / MEM_DIM;

pub struct Sam21 {
    encoder: Session,
    decoder: Session,
    memory_attention: Session,
    memory_encoder: Session,
    pointer_tpos: Session,
    /// `[7, 64]`: row 6 for the seed, row k − 1 for a frame k back.
    temporal: Vec<Vec<f32>>,
    total_frames: usize,
    frame: usize,
    seed: Option<Memory>,
    /// The most recent frames' memories, newest first.
    recent: VecDeque<Memory>,
}

struct Encoded {
    feats0: Array,
    feats1: Array,
    feats2: Array,
    feats2_no_mem: Array,
    position: Array,
}

impl Sam21 {
    pub fn load(dir: &Path, device: Device, total_frames: usize) -> Result<Self> {
        let constants: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("constants.json")).context("constants.json")?)?;
        let temporal: Vec<Vec<f32>> = serde_json::from_value(constants["memory_temporal_positional_encoding"].clone())
            .context("memory_temporal_positional_encoding")?;
        anyhow::ensure!(temporal.len() == BLOCKS && temporal.iter().all(|r| r.len() == MEM_DIM), "expected 7×64 temporal rows");
        let s = |name: &str| onnx::session(&dir.join(format!("{name}.onnx")), device);
        Ok(Self {
            encoder: s("vision_encoder")?,
            decoder: s("mask_decoder")?,
            memory_attention: s("memory_attention")?,
            memory_encoder: s("memory_encoder")?,
            pointer_tpos: s("pointer_tpos")?,
            temporal,
            total_frames,
            frame: 0,
            seed: None,
            recent: VecDeque::new(),
        })
    }

    fn encode(&mut self, pixel_values: &[f32]) -> Result<Encoded> {
        let input = Array::new([1, 3, 1024, 1024], pixel_values.to_vec());
        let mut out = onnx::run(
            &mut self.encoder,
            vec![("pixel_values", input.value()?)],
            &["feats0", "feats1", "feats2", "feats2_no_mem", "vision_pos_embed"],
        )?
        .into_iter();
        let mut next = || out.next().unwrap();
        Ok(Encoded { feats0: next(), feats1: next(), feats2: next(), feats2_no_mem: next(), position: next() })
    }

    fn decode(&mut self, enc: &Encoded, features: &Array, points: &[[f32; 2]], labels: &[i32]) -> Result<(Array, f32, Vec<f32>)> {
        let n = points.len();
        let coords = Array::new([1, 1, n, 2], points.iter().flatten().copied().collect());
        let out = onnx::run(
            &mut self.decoder,
            vec![
                ("feats0", enc.feats0.value()?),
                ("feats1", enc.feats1.value()?),
                ("feats2_cond", features.value()?),
                ("input_points", coords.value()?),
                ("input_labels", onnx::i32_value(vec![1, 1, n], labels.to_vec())?),
            ],
            &["high_res_mask", "object_score_logits", "object_pointer"],
        )?;
        let [mask, score, pointer] = out.try_into().unwrap();
        Ok((mask, score.data[0], pointer.data))
    }

    fn remember(&mut self, enc: &Encoded, mask: &Array, score: f32, binarize: bool, pointer: Vec<f32>) -> Result<Memory> {
        let out = onnx::run(
            &mut self.memory_encoder,
            vec![
                ("feats2", enc.feats2.value()?),
                ("high_res_mask", mask.clone().reshaped([1, 1, 1024, 1024]).value()?),
                ("object_score_logits", Array::new([1, 1], vec![score]).value()?),
                ("binarize", Array::new(Vec::<usize>::new(), vec![if binarize { 1.0 } else { 0.0 }]).value()?),
            ],
            &["memory_tokens", "memory_pos"],
        )?;
        let [features, positions] = out.try_into().unwrap();
        Ok(Memory { features: features.data, positions: positions.data, pointer })
    }

    /// The memory attention's `memory` and `memory_pos`: spatial blocks (seed, then 6…1 frames back, padded with the
    /// most recent), then 16 pointers split in 4 tokens each (seed, then 1…15 back, padded the same way).
    fn bank(&mut self) -> Result<(Array, Array)> {
        let seed = self.seed.as_ref().context("track before seed")?;
        let mut blocks: Vec<(&Memory, usize)> = vec![(seed, BLOCKS - 1)];
        for back in (1..=self.recent.len().min(BLOCKS - 1)).rev() {
            blocks.push((&self.recent[back - 1], back - 1));
        }
        let pad = *blocks.last().unwrap();
        blocks.resize(BLOCKS, pad);

        let max_pointers = self.total_frames.clamp(2, POINTERS);
        let mut pointers: Vec<(&[f32], f32)> = vec![(&seed.pointer, self.frame as f32)];
        for back in 1..max_pointers {
            match self.recent.get(back - 1) {
                Some(m) if back < self.frame => pointers.push((&m.pointer, back as f32)),
                _ => break,
            }
        }
        let last = *pointers.last().unwrap();
        pointers.resize(POINTERS, last);

        let diffs: Vec<f32> = pointers.iter().map(|(_, d)| d / (max_pointers - 1) as f32).collect();
        let tpos = onnx::run(
            &mut self.pointer_tpos,
            vec![("normalized_diffs", Array::new([POINTERS], diffs).value()?)],
            &["pointer_pos"],
        )?
        .remove(0);

        let len = BLOCKS * TOKENS + POINTERS * SPLITS;
        let (mut memory, mut position) = (Vec::with_capacity(len * MEM_DIM), Vec::with_capacity(len * MEM_DIM));
        for (m, row) in &blocks {
            memory.extend_from_slice(&m.features);
            let t = &self.temporal[*row];
            position.extend(m.positions.chunks_exact(MEM_DIM).flat_map(|p| p.iter().zip(t).map(|(a, b)| a + b)));
        }
        for (i, (pointer, _)) in pointers.iter().enumerate() {
            memory.extend_from_slice(pointer);
            for _ in 0..SPLITS {
                position.extend_from_slice(&tpos.data[i * MEM_DIM..][..MEM_DIM]);
            }
        }
        Ok((Array::new([len, 1, MEM_DIM], memory), Array::new([len, 1, MEM_DIM], position)))
    }
}

impl VideoTracker for Sam21 {
    fn seed(&mut self, rgb: &[u8], width: usize, height: usize, points: &[[f32; 2]], labels: &[i32]) -> Result<Step> {
        let enc = self.encode(&frame::pixel_values(rgb, width, height))?;
        let points: Vec<[f32; 2]> = points.iter().map(|p| frame::to_model_space(p[0], p[1], width, height)).collect();
        let points = &points[..];
        let (mask, score, pointer) = self.decode(&enc, &enc.feats2_no_mem, points, labels)?;
        self.seed = Some(self.remember(&enc, &mask, score, true, pointer)?);
        self.recent.clear();
        self.frame = 0;
        Ok(Step { mask: frame::binary_mask(&mask.data, width, height), object_score: score })
    }

    fn track(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Step> {
        self.frame += 1;
        let enc = self.encode(&frame::pixel_values(rgb, width, height))?;
        let (memory, memory_pos) = self.bank()?;
        let conditioned = onnx::run(
            &mut self.memory_attention,
            vec![
                ("current_vision_features", onnx::to_sequence(&enc.feats2).value()?),
                ("current_vision_position_embeddings", onnx::to_sequence(&enc.position).value()?),
                ("memory", memory.value()?),
                ("memory_pos", memory_pos.value()?),
            ],
            &["conditioned_feats"],
        )?
        .remove(0)
        .reshaped([1, 256, 64, 64]);
        let (mask, score, pointer) = self.decode(&enc, &conditioned, &[[0.0, 0.0]], &[-1])?;
        let memory = self.remember(&enc, &mask, score, false, pointer)?;
        self.recent.push_front(memory);
        self.recent.truncate(POINTERS - 1);
        Ok(Step { mask: frame::binary_mask(&mask.data, width, height), object_score: score })
    }
}
