#![recursion_limit = "512"]

use std::path::Path;

use burn::{
    backend::{Autodiff, Wgpu},
    module::Module,
    optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig},
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    train::metric::Adaptor,
};
use data::data_loaders::{CovidDataLoader, DataLoader};
use log::LevelFilter;
use log4rs::{
    append::file::FileAppender,
    config::{Appender, Root},
    encode::pattern::PatternEncoder,
    Config,
};
use model::model::VisionModelConfig;
use save::{load_from_highest, save_to_new_highest};
use train::{train::TrainingConfig, TrainingManager};
use visualization::display_inference;
pub mod data;
pub mod model;
pub mod save;
pub mod train;
pub mod visualization;
pub mod metric;

fn main() {
    if cfg!(feature = "debug_log") {
        println!("Logging");
        let logfile = FileAppender::builder()
            .encoder(Box::new(PatternEncoder::new("{l} - {m}\n")))
            .build("log/output.log")
            .unwrap();

        let config = Config::builder()
            .appender(Appender::builder().build("logfile", Box::new(logfile)))
            .build(Root::builder().appender("logfile").build(LevelFilter::Info))
            .unwrap();

        log4rs::init_config(config).unwrap();
    }

    display_inference();
    // train();
}

fn train() {
    log::info!("Hello, world!");
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = Default::default();


    let model_name = "adaptive_goal_shrunk_model";

    let training_config = TrainingConfig::new(model_name.to_string());

    let model = VisionModelConfig::new(3).init(&device)/* .load_record(record) */;
    let model = load_from_highest(model_name, model, &device);
    // save_to_new_highest(model_name, &model);

    let mut training_manager = TrainingManager::<MyAutodiffBackend>::init(training_config, device);

    let data_loader = CovidDataLoader::new_and_assert(&model);

    let model = training_manager.train_all(model, data_loader);
}
