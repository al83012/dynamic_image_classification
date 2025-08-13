use burn::{backend::Wgpu, prelude::*};
use nannou::{color::{BLACK, WHITE},  wgpu::{self, Texture}, App, Frame};

use crate::{
    infer::{self, StepInfo},
    model::{VisionModel, VisionModelConfig},
    save::load_from_highest,
};

pub struct DisplayData<B: Backend> {
    pub steps: Vec<StepInfo<B>>,
    pub texture: nannou::wgpu::Texture,
}

pub fn display_inference() {
    nannou::app(display_data_model).run();
}

// Path rel to data/ folder
pub fn display_data_model(app: &App) -> DisplayData<Wgpu<f32, i32>> {
    type MyBackend = Wgpu<f32, i32>;

    app.new_window()
        .size(512, 512)
        .view(view::<Wgpu<f32, i32>>)
        .build()
        .unwrap();

    let image_path = "covid/Covid19-dataset/test/Covid/COVID-00033.jpg";

    let device = Default::default();

    let model_name = "tests";

    let model :VisionModel<MyBackend> = VisionModelConfig::new(3).init(&device)/* .load_record(record) */;
    let model = load_from_highest(model_name, model, &device);

    let steps = infer::steps_to_finish(image_path, &model, &device, 50, 0.8);

    let image = nannou::image::open(format!("data/{image_path}")).expect("Failed to load image");
    let rgb_image = image.to_rgb8();
    let dyn_image = nannou::image::DynamicImage::ImageRgb8(rgb_image);
    let texture = Texture::from_image(app, &dyn_image);

    DisplayData { steps, texture }
}

pub fn view<B: Backend>(app: &App, model: &DisplayData<B>, frame: Frame) {
    let draw = app.draw();
    draw.background().color(WHITE);
    draw.texture(&model.texture).wh(app.window_rect().wh());

    draw.text(&format!("{}", model.steps.len()));

    draw.to_frame(app, &frame).unwrap();
}
