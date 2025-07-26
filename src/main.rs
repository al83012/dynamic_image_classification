use burn::{
    backend::{Autodiff, Wgpu},
    optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig},
    train::metric::Adaptor,
};
use data::{CovidDataLoader, DataLoader, SmokerDataLoader};
use log::LevelFilter;
use log4rs::{append::file::FileAppender, config::{Appender, Root}, encode::pattern::PatternEncoder, Config};
use model::VisionModelConfig;
use ppo::{train_all, OptimizerData};

pub mod class;
pub mod data;
pub mod image;
pub mod metrics;
pub mod model;
pub mod ppo;

fn main() {
    let logfile = FileAppender::builder()
        .encoder(Box::new(PatternEncoder::new("{l} - {m}\n")))
        .build("log/output.log").unwrap();

    let config = Config::builder()
        .appender(Appender::builder().build("logfile", Box::new(logfile)))
        .build(Root::builder()
                   .appender("logfile")
                   .build(LevelFilter::Info)).unwrap();

    log4rs::init_config(config).unwrap();

    log::info!("Hello, world!");
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = Default::default();

    let mut model = VisionModelConfig::new(3).init(&device);
    let mut data_loader = CovidDataLoader::new_and_assert(&model);
    let mut optim_data = OptimizerData::<MyAutodiffBackend> {
        class_optim: AdamConfig::new().init(),
        pos_optim: AdamConfig::new().init(),
    };

    model = train_all::<MyAutodiffBackend>(model, &device, &mut optim_data, data_loader, 100);
}
