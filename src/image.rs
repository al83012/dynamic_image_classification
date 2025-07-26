use burn::prelude::*;
use nn::pool::AdaptiveAvgPool2d;
use log;

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

    let (x0, x1) = if px < section_width / 2 {
        // Ends in the negative
        (0, section_width)
    } else if px + section_width / 2 > w - 1 {
        // Out of bounds
        (w - 1 - section_width, w - 1)
    } else {
        (px - section_width / 2, px + section_width / 2)
    };

    let (y0, y1) = if py < section_height / 2 {
        // Ends in the negative
        (0, section_height)
    } else if py + section_height / 2 > h - 1 {
        // Out of bounds
        (h - 1 - section_height, h - 1)
    } else {
        (py - section_height / 2, py + section_height / 2)
    };

    //slice all three channels and the block of x and y coords

    if y0 == 744 {
        log::info!("It happened: \ncx{cx}, cy{cy}, square_rel_size{square_rel_size}, shape{:?}", image.shape())
    }
    let slice = image.slice([0..3, x0..x1, y0..y1]);
    if y0 == 744 {
        log::info!("Still here");
    }
    slice

}
