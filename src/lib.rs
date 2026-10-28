//! Promptable video object tracking: one click on a seed frame, then the object frame by frame with a memory bank.
//!
//! Two exports of the same family run behind one trait: EdgeTAM (`jax-image-tools/edgetam-video-onnx`) and SAM 2.1
//! tiny (`square-zero-labs/sam2.1-tiny-video-onnx`). Both encode 1024² frames and keep the bank on the host, as
//! transformers' `*VideoModel` does: the seed frame's memory plus the six most recent frames', and up to 16 object
//! pointers.

pub mod cuda;
pub mod edgetam;
pub mod frame;
pub mod onnx;
pub mod sam21;

pub use onnx::Device;

/// One frame's result.
#[derive(Debug, Clone)]
pub struct Step {
    /// The object at the video's size, row-major, 1 where it is.
    pub mask: Vec<u8>,
    /// > 0 when the model sees the object on this frame.
    pub object_score: f32,
}

/// Frames are decoded RGB, 8-bit, interleaved, at the video's size.
pub trait VideoTracker {
    /// The prompted frame. `points` are in the video's pixels; labels 1 positive, 0 negative, 2 and 3 a box's
    /// corners.
    fn seed(&mut self, rgb: &[u8], width: usize, height: usize, points: &[[f32; 2]], labels: &[i32]) -> anyhow::Result<Step>;

    /// The next frame after the seed or the last tracked one.
    fn track(&mut self, rgb: &[u8], width: usize, height: usize) -> anyhow::Result<Step>;
}

/// A tracked frame's memory, kept on the host by the SAM 2.1 driver.
#[derive(Debug, Clone)]
struct Memory {
    features: Vec<f32>,
    positions: Vec<f32>,
    pointer: Vec<f32>,
}
