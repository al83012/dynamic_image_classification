use burn::backend::Wgpu;
use model::VisionModelConfig;

pub mod model;

fn main() {
    type MyBackend = Wgpu<f32, i32>;

    let device: MyBackend = Default::default();
    let model = VisionModelConfig::new(10);

    println!("{}", model);
}
