use burn::{
    data::dataloader::Progress,
    optim::{
        adaptor::OptimizerAdaptor, Adam, AdamState, GradientsParams, Optimizer, SimpleOptimizer,
    },
    prelude::Backend,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
};
use image::GenericImageView;
use std::{clone::Clone, fs};

use burn_train::{
    metric::MetricEntry,
    renderer::{tui::TuiMetricsRenderer, MetricState, MetricsRenderer, TrainingProgress},
    SummaryMetrics, TrainingInterrupter,
};

use rand::seq::SliceRandom;
use rand::thread_rng;

use burn::prelude::*;
use nn::{
    loss::{MseLoss, Reduction},
    LstmState,
};

use crate::{
    data::ClassificationItem,
    image::{extract_section, load_image},
    model::{PositioningData, VisionModel, VisionModelStepInput},
};

pub struct OptimizerData<B: AutodiffBackend> {
    pub class_optim: OptimizerAdaptor<Adam, VisionModel<B>, B>,
    pub pos_optim: OptimizerAdaptor<Adam, VisionModel<B>, B>,
}

#[derive(Debug, Clone, Copy)]
pub struct StepStatistics {
    pub reward: f32,
    pub last_loss: f32,
    pub avg_loss: f32,
    pub finished_after: usize,
}

pub fn train<B: AutodiffBackend>(
    data: ClassificationItem<B>,
    model: VisionModel<B>,
    device: &B::Device,
    max_iter_count: usize,
    optim_data: &mut OptimizerData<B>,
) -> (VisionModel<B>, StepStatistics) {
    let mut model = model;

    let class_optim = &mut optim_data.class_optim;
    let pos_optim = &mut optim_data.pos_optim;

    let image_tensor = data.image;
    let target = data.classification as usize;

    // let idx_tensor: Tensor<B, 1, Int> = Tensor::from_data([target], device);
    // let class_oh_target_int: Tensor<B, 1, Int> = idx_tensor.one_hot(model.num_classes);
    // let class_oh_target: Tensor<B, 3> = class_oh_target_int.float().unsqueeze_dims(&[0, 0]);
    // let class_oh_target_vec: Vec<f32> = class_oh_target.to_data().to_vec().unwrap();
    let class_oh_target: Tensor<B, 3> =
        Tensor::<B, 1>::from_data(if target == 0 { [1.0, 0.0] } else { [0.0, 1.0] }, device)
            .unsqueeze_dims(&[0, 1]);

    // println!("Extracted class data");

    let pos_data = PositioningData::<B>::start(device);
    let mut lstm_state: Option<LstmState<B, 2>> = None;

    let mut pos_out_dummy_diff_acc: Tensor<B, 1> = Tensor::from_data([0.0], device);

    let mse_loss = MseLoss::new();

    let mut acc_reward = 0.0;

    let mut current_iter = 0;

    let mut correct_output = false;

    let mut last_loss = 0.0;

    let mut aggregate_loss = 0.0;

    // println!("Setup for iteration");

    for i in 0..max_iter_count {
        current_iter = i;
        // println!("Iter[{current_iter:?}]");
        let ([cx, cy], rel_size) = pos_data.get_params_detach();
        let image_section = extract_section(image_tensor.clone(), cx, cy, rel_size);
        let step_in = VisionModelStepInput {
            image_section,
            pos_data: pos_data.clone(),
            lstm_state,
        };

        // println!("Before fwd");
        let step_out = model.forward(step_in);

        // println!("After fwd");
        lstm_state = Some(step_out.next_lstm_state);

        let class_out = step_out.current_classification;
        let pos_out = step_out.next_pos.0;

        let (highest_class, certainty) =
            tensor_argmax(class_out.clone().detach().squeeze_dims(&[0, 1]));

        let output_vec: Vec<f32> = class_out
            .clone()
            .detach()
            .squeeze_dims::<1>(&[0, 1])
            .to_data()
            .to_vec()
            .expect("Error unwrapping output");
        assert_eq!(output_vec.len(), 2);

        let min = output_vec[0].min(output_vec[1]);
        let max = output_vec[0].max(output_vec[1]);

        let can_finish = min < 0.5 && max >= 0.5;

        // println!("After argmax");

        let class_loss = mse_loss.forward(class_out, class_oh_target.clone(), Reduction::Mean);
        let class_loss_single: f32 = class_loss.clone().detach().into_data().to_vec().unwrap()[0];

        aggregate_loss += class_loss_single;

        // println!("After loss");

        let class_grad = class_loss.backward();
        let class_grad_params = GradientsParams::from_grads(class_grad, &model);

        let pos_out_dummy_diff = pos_out.mean();
        pos_out_dummy_diff_acc = Tensor::cat(vec![pos_out_dummy_diff_acc, pos_out_dummy_diff], 0);

        model = class_optim.step(model.class_lr, model, class_grad_params);

        if min < 0.0 || max > 1.0 {
            acc_reward -= 0.5;
        }

        if can_finish {
            if highest_class == target {
                correct_output = true;
            }
            last_loss = class_loss.detach().to_data().to_vec().unwrap()[0];
            acc_reward -= class_loss_single;
            break;
        }

        acc_reward += class_loss_single;
    }

    aggregate_loss /= (current_iter + 1) as f32;

    acc_reward += if correct_output { 10.0 } else { -5.0 };

    if last_loss > aggregate_loss {
        acc_reward += (last_loss - aggregate_loss) * 5.0 + 1.0;
    }

    let time_needed = (current_iter + 1) as f32 / max_iter_count as f32;

    // println!("Time: {time_needed:.2}, right = {correct_output}");

    // If the answer is correct, it is better if the time is longer. If the answer is wrong, it is
    // better to see that only a short time was taken
    // let time_goal = if correct_output {
    //     1.0 - time_needed
    // } else {
    //     time_needed
    // };
    let time_goal = (1.0 - time_needed.sqrt());

    let pos_out_dummy_diff_mean = pos_out_dummy_diff_acc.mean();
    let pos_dummy_loss = pos_out_dummy_diff_mean.mul_scalar(acc_reward * time_goal);
    let pos_dummy_grad = pos_dummy_loss.backward();
    let pos_dummy_grad_params = GradientsParams::from_grads(pos_dummy_grad, &model);

    model = pos_optim.step(model.pos_lr, model, pos_dummy_grad_params);

    (
        model,
        StepStatistics {
            reward: acc_reward * time_goal,
            last_loss,
            avg_loss: aggregate_loss,
            finished_after: current_iter + 1,
        },
    )
}

fn tensor_argmax<B: Backend>(t: Tensor<B, 1>) -> (usize, f32) {
    let vec = t.into_data().to_vec::<f32>().unwrap();
    vec.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(a, b)| (a, *b))
        .unwrap()
}

pub fn train_all<B: AutodiffBackend>(
    mut model: VisionModel<B>,
    device: &B::Device,
    optim_data: &mut OptimizerData<B>,
    epochs: usize,
) -> VisionModel<B> {
    let train_interrupter = TrainingInterrupter::new();
    let mut train_iter_metric = MetricState::Numeric(
        MetricEntry {
            name: "Iteration Number".to_string(),
            formatted: "Iter Num".to_string(),
            serialize: "itr_num".to_string(),
        },
        0.0,
    );
    let mut renderer = TuiMetricsRenderer::new(train_interrupter, None);
    // println!("loading training folder");
    let training_folder =
        fs::read_dir("./data/archive/Training/Training").expect("Unable to open Training Data");
    let mut entries = training_folder.into_iter().collect::<Vec<_>>();
    let mut rng = rand::thread_rng();

    let total_entries = entries.len();
    let mut processed = 0;

    entries.shuffle(&mut rng);

    for epoch in 0..epochs {
        // println!("processing entries");
        for chunk in entries.chunks(50) {
            for entry in chunk {
                processed += 1;

                let entry = entry.as_ref().expect("Unable to read dir entry");
                let file_name = entry.file_name().to_string_lossy().to_string();
                // println!("Loading {file_name}");
                let class = if file_name.starts_with("not") { 0 } else { 1 };

                let image = load_image::<B>(&format!("Training/Training/{file_name}"), device);

                let training_input = ClassificationItem {
                    image,
                    classification: class,
                };

                let (model_out, stats) = train(training_input, model, device, 30, optim_data);
                model = model_out;

                let new_iter_metric = MetricState::Numeric(
                    MetricEntry {
                        name: "Iteration Number".to_string(),
                        formatted: "Iter Num".to_string(),
                        serialize: "itr_num".to_string(),
                    },
                    stats.finished_after as f64,
                );

                let new_reward_metric = MetricState::Numeric(
                    MetricEntry {
                        name: "Reward".to_string(),
                        formatted: "Reward".to_string(),
                        serialize: "reward".to_string(),
                    },
                    stats.reward as f64,
                );

                let new_last_loss_metric = MetricState::Numeric(
                    MetricEntry {
                        name: "Last Loss".to_string(),
                        formatted: "Last Loss".to_string(),
                        serialize: "last_loss".to_string(),
                    },
                    stats.last_loss as f64,
                );

                let new_avg_loss_metric = MetricState::Numeric(
                    MetricEntry {
                        name: "Avg Loss".to_string(),
                        formatted: "Avg Loss".to_string(),
                        serialize: "avg_loss".to_string(),
                    },
                    stats.avg_loss as f64,
                );
                renderer.update_train(new_iter_metric);
                renderer.update_train(new_reward_metric);
                renderer.update_train(new_last_loss_metric);
                renderer.update_train(new_avg_loss_metric);
                let training_progress = TrainingProgress {
                    progress: Progress::new(processed, total_entries),
                    epoch: epoch + 1,
                    epoch_total: epochs,
                    iteration: 1,
                };
                renderer.render_train(training_progress);

                // println!("{stats:#?}");
            }

            create_artifact_dir("model_artifacts");
            model
                .clone()
                .save_file("model_artifacts", &CompactRecorder::new());
        }
    }

    model
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}
