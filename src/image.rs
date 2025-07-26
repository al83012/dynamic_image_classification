use burn::prelude::*;
use nn::pool::AdaptiveAvgPool2d;

use crate::data::ClassificationBatch;

// Tensor[channels, x, y]
pub const IMAGE_FOLDER: &str = "data";
pub fn load_image<B: Backend>(name: &str, device: &B::Device) -> Tensor<B, 3> {
    let img = image::open(format!("{IMAGE_FOLDER}/{name}"))
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

    let reshaped_cx = cx.atan() / (std::f32::consts::PI / 2.0) * 0.5;
    let reshaped_cy = cy.atan() / (std::f32::consts::PI / 2.0) * 0.5;

    let px = ((reshaped_cx.clamp(-0.5, 0.5) + 0.5) * w as f32).round() as usize;
    let py = ((reshaped_cy.clamp(-0.5, 0.5) + 0.5) * h as f32).round() as usize;

    let clamped_size = square_rel_size.clamp(0.1, 0.9);

    let section_width = (w as f32 * clamped_size) as usize;
    let section_height = (h as f32 * clamped_size) as usize;

    let desired_x0 = px.saturating_sub(section_width / 2);
    let desired_y0 = py.saturating_sub(section_height / 2);

    let desired_x1 = (desired_x0 + section_width).min(w - 2);
    let desired_y1 = (desired_y0 + section_height).min(h - 2);

    let desired_x0 = (desired_x1 - section_width).saturating_sub(1);
    let desired_y0 = (desired_y1 - section_height).saturating_sub(1);



    //slice all three channels and the block of x and y coords
    image.slice([0..3, desired_x0..desired_x1, desired_y0..desired_y1])
}
