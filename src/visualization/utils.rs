use burn::prelude::*;
use nannou::geom::{Rect, Vector2};

use crate::{model::PositioningData, train::utils::smoothstep};

pub struct RelImageRect {
    pub center_pos: [f32; 2],
    pub rel_size: f32,
}

pub fn rel_rect_from_pos<B: Backend>(pos_data: &PositioningData<B>) -> RelImageRect {
    let data = pos_data.get_params_detach();
    let ([cx, cy], square_rel_size) = data;

    let reshaped_cx = (cx.atan() / (std::f32::consts::PI / 2.0) * 0.5).clamp(-0.5, 0.5) + 0.5;
    let reshaped_cy = (cy.atan() / (std::f32::consts::PI / 2.0) * 0.5).clamp(-0.5, 0.5) + 0.5;

    RelImageRect {
        center_pos: [reshaped_cx, reshaped_cy],
        rel_size: square_rel_size.clamp(0.1, 0.9),
    }
}

impl RelImageRect {
    pub fn lerp(&self, other: &Self, t: f32) -> RelImageRect {
        let t = t.clamp(0.0, 1.0);

        let [cx1, cy1] = self.center_pos;
        let [cx2, cy2] = other.center_pos;

        let rel_size_1 = self.rel_size;
        let rel_size_2 = other.rel_size;

        let cx = cx1 * (1.0 - t) + cx2 * t;
        let cy = cy1 * (1.0 - t) + cy2 * t;
        let rel_size = rel_size_1 * (1.0 - t) + rel_size_2 * t;

        Self {
            center_pos: [cx, cy],
            rel_size,
        }
    }

    pub fn in_context(&self, image_rect: Rect) -> Rect {
        let w = self.rel_size * image_rect.w();
        let h = self.rel_size * image_rect.h();

        let wh = Vector2::new(w, h);

        let x_range = [
            image_rect.x() - image_rect.w() / 2.0,
            image_rect.x() + image_rect.w() / 2.0,
        ];
        let y_range = [
            image_rect.y() - image_rect.h() / 2.0,
            image_rect.y() + image_rect.h() / 2.0,
        ];

        let [cx, cy] = self.center_pos;

        let adj_center = Vector2::new(
            x_range[0] * (1.0 - cx) + x_range[1] * cx,
            y_range[0] * (1.0 - cy) + y_range[1] * cy,
        );

        Rect::from_xy_wh(adj_center, wh)
    }
}

// remapping a steady slope into a jagged one
pub fn remap_t(start: f32, end: f32, t: f32) -> f32 {
    let fract = t.fract();
    let whole = t - fract;
    let start = start.clamp(0.0, 1.0);
    let end = end.clamp(start, 1.0);

    let len = end - start;

    let x = (fract - start) / len;
    x.clamp(0.0, 1.0) + whole
}


pub fn smooth_remap_t(start: f32, end: f32, t: f32) -> f32 {
    let t_remap = remap_t(start, end, t);

    let fract = t_remap.fract();
    let whole = t_remap - fract;

    smoothstep(fract) + whole
}
