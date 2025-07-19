use burn::prelude::*;
use nn::pool::AdaptiveAvgPool2d;

use crate::data::ClassificationBatch;

// Tensor[channels, x, y]
pub const IMAGE_FOLDER: &str = "data/archive/";
pub fn load_image<B: Backend>(name: &str, device: &B::Device) -> Tensor<B, 3> {
    let img = image::open(format!("{IMAGE_FOLDER}/{name}.png"))
        .expect("Error loading image")
        .to_rgb8();
    let (w, h) = img.dimensions();
    let arr: Vec<f32> = img
        .pixels()
        .flat_map(|p| {
            let [r, g, b] = p.0;
            [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]
        })
        .collect();
    let image_shape = [3, w as usize, h as usize];
    Tensor::from_data(TensorData::new(arr, image_shape), device)
}

pub fn extract_section<B: Backend>(
    image: Tensor<B, 3>,
    cx: f32,
    cy: f32,
    square_rel_size: f32,
) -> Tensor<B, 3> {
    let [c, h, w] = image.dims();
    let px = (cx.clamp(0.0, 1.0) * w as f32).round() as usize;
    let py = (cy.clamp(0.0, 1.0) * h as f32).round() as usize;

    let half_section_width: usize =
        (square_rel_size.clamp(0.1, 1.0) / 2.0 * w as f32).round() as usize;
    let half_section_height: usize =
        (square_rel_size.clamp(0.1, 1.0) / 2.0 * h as f32).round() as usize;

    let x0 = px.saturating_sub(half_section_width);
    let y0 = py.saturating_sub(half_section_height);

    let x1 = px + half_section_width.clamp(0, w - 1);
    let y1 = (px + half_section_height).clamp(0, h - 1);

    //slice all three channels and the block of x and y coords
    image.slice([0..3, x0..x1, y0..y1])
}
