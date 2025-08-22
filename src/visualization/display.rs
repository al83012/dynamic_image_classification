use burn::{backend::Wgpu, prelude::*};
use nannou::{
    color::{DARKBLUE, GREY},
    draw::theme::Color,
    geom::{self, rect, Padding, Rect},
    image::{GenericImageView, ImageBuffer},
    ui::color::DARK_GREY,
    wgpu::{self, Texture},
    App, Draw, Frame,
};

use crate::{
    data::image::image_as_tensor,
    model::{model::VisionModelConfig, VisionModel},
    save::load_from_highest,
    train::utils::smoothstep,
    visualization::infer,
};

use super::{
    infer::StepInfo,
    utils::{rel_rect_from_pos, remap_t, smooth_remap_t},
};

pub struct DisplayData<B: Backend> {
    pub steps: Vec<StepInfo<B>>,
    pub texture: nannou::wgpu::Texture,
    pub image: Tensor<B, 3>,
    pub panel_rect: Rect,
    pub img_rect: Rect,
}

pub fn display_inference() {
    nannou::app(display_data_model).run();
}

// Path rel to data/ folder
pub fn display_data_model(app: &App) -> DisplayData<Wgpu<f32, i32>> {
    type MyBackend = Wgpu<f32, i32>;

    app.new_window()
        .size(1920, 1080)
        .maximized(true)
        .view(view::<Wgpu<f32, i32>>)
        .build()
        .unwrap();

    let image_path = "covid/Covid19-dataset/train/Viral Pneumonia/Gen-10.jpg";

    let device = Default::default();

    let model_name = "adaptive_goal_shrunk_model";

    let model :VisionModel<MyBackend> = VisionModelConfig::new(3).init(&device)/* .load_record(record) */;
    let model = load_from_highest(model_name, model, &device);

    let steps = infer::steps_to_finish(image_path, &model, &device, 200, 0.8);

    let image = nannou::image::open(format!("data/{image_path}")).expect("Failed to load image");

    let rgb_image = image.to_rgb8();
    let dyn_image = nannou::image::DynamicImage::ImageRgb8(rgb_image);
    let image_tensor = image_as_tensor(&dyn_image, &device);
    let texture = Texture::from_image(app, &dyn_image);

    let (img_w, img_h) = dyn_image.dimensions();

    let panel_rect = panel_rect(app);
    let img_rect = image_rect(app, img_w, img_h);

    DisplayData {
        steps,
        texture,
        image: image_tensor,
        panel_rect,
        img_rect,
    }
}

pub fn view<B: Backend>(app: &App, model: &DisplayData<B>, frame: Frame) {
    let draw = app.draw();
    draw.background()
        .rgb(36.0 / 255.0, 41.0 / 255.0, 46.0 / 255.0);
    // draw.text(&format!("{}", model.steps.len()));
    draw.to_frame(app, &frame).unwrap();

    draw_image(app, model, &frame);
    draw_panel(app, model, &frame);
}

pub fn draw_image<B: Backend>(app: &App, model: &DisplayData<B>, frame: &Frame) {
    let img_rect = model.img_rect;
    let image = &model.texture;

    let draw = app.draw();
    draw.texture(image).xy(img_rect.xy()).wh(img_rect.wh());
    draw.to_frame(app, frame).expect("Error drawing image");
}

pub fn draw_panel<B: Backend>(app: &App, model: &DisplayData<B>, frame: &Frame) {
    let panel_rect = model.panel_rect;
    let draw = app.draw();
    let info_rect = Rect::from_w_h(panel_rect.w(), 50.0)
        .align_top_of(panel_rect)
        .align_middle_x_of(panel_rect);
    let time = app.time;
    let output_time = smooth_remap_t(0.4, 1.0, time);
    let focus_time = smooth_remap_t(0.0, 0.5, time);
    draw.text(&format!("Time: {time:.2}"))
        .wh(info_rect.wh())
        .xy(info_rect.xy())
        .rgb8(62, 128, 224);
    draw.to_frame(app, frame).unwrap();
    draw_focus(app, frame, model, focus_time);
    draw_outputs(app, model, output_time, frame);
}

pub fn panel_rect(app: &App) -> Rect {
    let window_rect = app.window_rect().pad(50.0);
    let (w_w, w_h) = window_rect.w_h();
    Rect::from_w_h(w_w * 0.25, w_h)
        .align_right_of(window_rect)
        .align_middle_y_of(window_rect)
        .pad(20.0)
}

pub fn image_rect(app: &App, width: u32, height: u32) -> Rect {
    let window_rect = app.window_rect().pad(50.0);

    let (w_w, w_h) = window_rect.w_h();
    // println!("Window: ({w_w}, {w_h})");

    //Maximum available dimensions to image
    let img_fill_w = w_w * 0.75;
    let img_fill_h = w_h;

    // println!("Img Fill: ({img_fill_w}, {img_fill_h})");

    let fill_ratio = img_fill_w / img_fill_h;
    let img_ratio = width as f32 / height as f32;

    // println!("Img Fill Ratio: {fill_ratio}");

    // println!("Img Ratio: {img_ratio}");

    let (img_w, img_h) = if img_ratio > fill_ratio {
        // Limited by width

        let width = img_fill_w;
        let height = width / img_ratio;
        (width, height)
    } else {
        let height = img_fill_h;
        let width = height * img_ratio;
        (width, height)
    };

    // println!("Img Adj Dim: ({img_w}, {img_h})");

    let img_rect = geom::Rect::from_w_h(img_w, img_h)
        .align_middle_y_of(window_rect)
        .align_left_of(window_rect);

    img_rect
}

pub fn draw_outputs<B: Backend>(app: &App, model: &DisplayData<B>, t: f32, frame: &Frame) {
    let full_idx = t.floor() as usize;

    let step_t = t.fract();

    // println!("At start T = {t}");

    let current_tensor: Vec<f32> = if (0.5 - step_t).abs() > 0.5 - 1e-2 {
        // println!("Full value");

        let id = if step_t > 0.5 {
            full_idx + 1
        } else {
            full_idx
        };

        let tensor = model.steps[id].class_out.clone();

        // println!("Tensor: {tensor:#?}");

        let tensor: Tensor<B, 1> = tensor.squeeze_dims(&[0, 1]);

        let tensor = burn::tensor::activation::softmax(tensor, 0);

        // println!("After Full value");

        tensor.to_data().to_vec().expect("Failed collecting to vec")
    } else {
        // println!("Lerped value");
        let from: Tensor<B, 1> = model.steps[full_idx]
            .class_out
            .clone()
            .squeeze_dims(&[0, 1]);
        let to: Tensor<B, 1> = model.steps[full_idx + 1]
            .class_out
            .clone()
            .squeeze_dims(&[0, 1]);

        let from = burn::tensor::activation::softmax(from, 0);
        let to = burn::tensor::activation::softmax(to, 0);

        from.to_data()
            .to_vec::<f32>()
            .expect("Failed collecting to vec")
            .into_iter()
            .zip(
                to.to_data()
                    .to_vec::<f32>()
                    .expect("Failed collecting to vec"),
            )
            .map(|(a, b)| a * (1.0 - step_t) + b * step_t)
            .collect()
    };

    let names = ["Covid", "Normal", "Pneumonia"];

    let value_section = model.panel_rect.pad_top(50.0);
    let value_height = 40.0;

    let full_value_width = value_section.w();

    let draw = app.draw();

    for (i, (name, val)) in names
        .into_iter()
        .zip(current_tensor.into_iter())
        .enumerate()
    {
        let offset = i as f32 * (value_height + 5.0);

        let rect_width = val * full_value_width;

        let val_rect = Rect::from_w_h(rect_width, value_height)
            .align_left_of(value_section)
            .align_top_of(value_section.pad_top(offset));

        draw.rect()
            .wh(val_rect.wh())
            .xy(val_rect.xy())
            .rgb8(48, 54, 60);

        let text_w = val_rect.w().max(40.0);
        let text_h = val_rect.h();

        let text_rect = Rect::from_w_h(text_w, text_h)
            .align_left_of(val_rect)
            .align_middle_y_of(val_rect)
            .shift_x(5.0);

        draw.text(name)
            .wh(text_rect.wh())
            .xy(text_rect.xy())
            .align_text_middle_y()
            .left_justify()
            .rgb(1.0, 1.0, 1.0);
    }

    draw.to_frame(app, frame).unwrap();
}

pub fn draw_focus<B: Backend>(app: &App, frame: &Frame, model: &DisplayData<B>, t: f32) {
    let img_rect = model.img_rect;

    let from_idx = t.floor() as usize;
    let fract = t.fract();

    let from_rect = rel_rect_from_pos(&model.steps.get(from_idx).unwrap().pos_in);
    let to_rect = rel_rect_from_pos(&model.steps.get(from_idx).unwrap().pos_out);

    let rect = from_rect.lerp(&to_rect, fract);
    let in_context = rect.in_context(img_rect);

    let draw = app.draw();

    draw.rect()
        .wh(in_context.wh())
        .xy(in_context.xy())
        .rgba8(150, 150, 150, 10)
        .stroke_weight(1.0)
        .stroke(DARKBLUE);

    draw.to_frame(app, frame);
}
