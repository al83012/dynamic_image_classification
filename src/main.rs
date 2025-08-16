#![recursion_limit = "512"]

use std::path::Path;

use burn::{
    backend::{Autodiff, Wgpu},
    module::Module,
    optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig},
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    train::metric::Adaptor,
};
use data::{CovidDataLoader, DataLoader, SmokerDataLoader};
use log::LevelFilter;
use log4rs::{
    append::file::FileAppender,
    config::{Appender, Root},
    encode::pattern::PatternEncoder,
    Config,
};
use model::{VisionModelConfig, VisionModelRecord};
use save::{load_from_highest, save_to_new_highest};
use train::{TrainingConfig, TrainingManager};
pub mod pos_opt;
pub mod modern_lstm;
pub mod class;
pub mod data;
pub mod display;
pub mod image;
pub mod infer;
pub mod metrics;
pub mod model;
pub mod save;
pub mod train;

fn main() {
    // let mut optim_data = OptimizerData::<MyAutodiffBackend> {
    //     class_optim: AdamConfig::new().init(),
    //     pos_optim: AdamConfig::new().init(),
    // };

    // model = train_all::<MyAutodiffBackend>(
    //     model,
    //     &device,
    //     &mut optim_data,
    //     data_loader,
    //     100,
    //     "model_artifacts-rerun1",
    // );

    // display::display_inference();
    train();
}

fn train() {
    if cfg!(feature = "debug_log") {
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

    log::info!("Hello, world!");
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = Default::default();

    // let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();

    // let artifact_path = Path::new("model_artifacts/model_artifacts.mpk");

    // let record = recorder
    //     .load(artifact_path.into(), &device)
    //     .expect("Error decoding state from specified path");

    let model_name = "combined_training_adj_goals";

    let training_config = TrainingConfig::new(model_name.to_string());

    let model = VisionModelConfig::new(3).init(&device)/* .load_record(record) */;
    let model = load_from_highest(model_name, model, &device);
    // save_to_new_highest(model_name, &model);

    let mut training_manager = TrainingManager::<MyAutodiffBackend>::init(training_config, device);

    let data_loader = CovidDataLoader::new_and_assert(&model);

    let model = training_manager.train_all(model, data_loader);
}
