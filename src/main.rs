use burn::{backend::{Autodiff, Wgpu}, optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig}, train::metric::Adaptor};
use data::{DataLoader, SmokerDataLoader, CovidDataLoader};
use model::VisionModelConfig;
use ppo::{train_all, OptimizerData};

pub mod model;
pub mod ppo;
pub mod image;
pub mod class;
pub mod data;
pub mod metrics;

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = Default::default();
    
    let mut model = VisionModelConfig::new(3).init(&device);
    let mut data_loader = CovidDataLoader::new_and_assert(&model);
    let mut optim_data = OptimizerData::<MyAutodiffBackend>{
        class_optim: AdamConfig::new().init(),
        pos_optim: AdamConfig::new().init(),
    };

    model = train_all::<MyAutodiffBackend>(model, &device, &mut optim_data, data_loader, 100);


}
