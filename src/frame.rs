//! What the trackers' vision encoders expect, and how their 1024² masks go back to the video's size.

/// The trackers' square input side.
pub const SIDE: usize = 1024;

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// An RGB frame (8-bit, row-major, interleaved) turned into the encoder's NCHW float input: resized to 1024²
/// without keeping the aspect ratio, scaled to 0–1 and normalised with ImageNet's mean and std, as transformers'
/// SAM 2 processor does.
pub fn pixel_values(rgb: &[u8], width: usize, height: usize) -> Vec<f32> {
    assert_eq!(rgb.len(), width * height * 3, "frame size");
    let resized = if (width, height) == (SIDE, SIDE) {
        rgb.to_vec()
    } else {
        let img = image::RgbImage::from_raw(width as u32, height as u32, rgb.to_vec()).expect("frame size");
        image::imageops::resize(&img, SIDE as u32, SIDE as u32, image::imageops::FilterType::Triangle).into_raw()
    };
    let plane = SIDE * SIDE;
    let mut out = vec![0f32; 3 * plane];
    for (i, px) in resized.chunks_exact(3).enumerate() {
        for c in 0..3 {
            out[c * plane + i] = (px[c] as f32 / 255.0 - MEAN[c]) / STD[c];
        }
    }
    out
}

/// A point in the video's pixels, in the 1024² space the prompt encoder works in.
pub fn to_model_space(x: f32, y: f32, width: usize, height: usize) -> [f32; 2] {
    [x * SIDE as f32 / width as f32, y * SIDE as f32 / height as f32]
}

/// Bilinear resize of a single-channel map, matching `F.interpolate(mode="bilinear", align_corners=False)`.
pub fn resize_bilinear(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    assert_eq!(src.len(), sw * sh);
    let (sx, sy) = (sw as f32 / dw as f32, sh as f32 / dh as f32);
    let axis = |d: usize, scale: f32, n: usize| {
        let s = ((d as f32 + 0.5) * scale - 0.5).max(0.0);
        let i0 = (s.floor() as usize).min(n - 1);
        let i1 = (i0 + 1).min(n - 1);
        (i0, i1, s - i0 as f32)
    };
    let cols: Vec<_> = (0..dw).map(|x| axis(x, sx, sw)).collect();
    let mut out = Vec::with_capacity(dw * dh);
    for y in 0..dh {
        let (y0, y1, fy) = axis(y, sy, sh);
        let (r0, r1) = (&src[y0 * sw..][..sw], &src[y1 * sw..][..sw]);
        for &(x0, x1, fx) in &cols {
            let top = r0[x0] + (r0[x1] - r0[x0]) * fx;
            let bottom = r1[x0] + (r1[x1] - r1[x0]) * fx;
            out.push(top + (bottom - top) * fy);
        }
    }
    out
}

/// Mask logits at the video's size, thresholded at 0.
pub fn binary_mask(logits_1024: &[f32], width: usize, height: usize) -> Vec<u8> {
    resize_bilinear(logits_1024, SIDE, SIDE, width, height).into_iter().map(|v| (v > 0.0) as u8).collect()
}

/// Intersection over union of two masks (non-zero is the object).
pub fn iou(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut inter, mut union) = (0u64, 0u64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x != 0, y != 0);
        inter += (x && y) as u64;
        union += (x || y) as u64;
    }
    if union == 0 { 1.0 } else { inter as f64 / union as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_matches_torch_half_pixel_centres() {
        // torch: F.interpolate(tensor([[[[0., 1.], [2., 3.]]]]), size=(4, 4), mode="bilinear", align_corners=False)
        let out = resize_bilinear(&[0.0, 1.0, 2.0, 3.0], 2, 2, 4, 4);
        let expected = [
            0.0, 0.25, 0.75, 1.0, 0.5, 0.75, 1.25, 1.5, 1.5, 1.75, 2.25, 2.5, 2.0, 2.25, 2.75, 3.0,
        ];
        for (a, b) in out.iter().zip(expected) {
            assert!((a - b).abs() < 1e-6, "{out:?}");
        }
    }

    #[test]
    fn normalisation_is_imagenet() {
        let px = pixel_values(&[255; SIDE * SIDE * 3], SIDE, SIDE);
        assert!((px[0] - (1.0 - 0.485) / 0.229).abs() < 1e-5);
        assert!((px[2 * SIDE * SIDE] - (1.0 - 0.406) / 0.225).abs() < 1e-5);
    }

    #[test]
    fn iou_of_empty_masks_is_one() {
        assert_eq!(iou(&[0, 0], &[0, 0]), 1.0);
        assert_eq!(iou(&[1, 0], &[1, 255]), 0.5);
    }
}
