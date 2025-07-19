use burn::{backend::{Autodiff, Wgpu}, optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig}, train::metric::Adaptor};
use model::VisionModelConfig;
use ppo::{train_all, OptimizerData};

pub mod model;
pub mod ppo;
pub mod image;
pub mod class;
pub mod data;

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = Default::default();
    
    let mut model = VisionModelConfig::new(10).init(&device);
    let mut optim_data = OptimizerData::<MyAutodiffBackend>{
        class_optim: AdamConfig::new().init(),
        pos_optim: AdamConfig::new().init(),
    };

    model = train_all::<MyAutodiffBackend>(model, &device, &mut optim_data);


}
