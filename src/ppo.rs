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
    data::{ClassificationItem, DataLoader},
    image::{extract_section, load_image},
    metrics::AvgMetric,
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
    pub correct: bool,
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

    let mut class_oh_vec = vec![0.0; model.num_classes];
    class_oh_vec[target] = 1.0;
    let class_oh_target: Tensor<B, 3> =
        Tensor::<B, 1>::from_data(class_oh_vec.as_slice(), device).unsqueeze_dims(&[0, 1]);

    // println!("Extracted class data");

    let mut pos_data = PositioningData::<B>::start(device);
    let mut lstm_state: Option<LstmState<B, 2>> = None;

    let mut pos_out_dummy_diff_acc: Tensor<B, 1> = Tensor::from_data([0.0], device);

    let mse_loss = MseLoss::new();

    let mut acc_reward = 0.0;

    let mut current_iter = 0;

    let mut correct_output = false;

    let mut last_loss = 0.0;

    let mut aggregate_loss = 0.0;

    let mut avg_norm_quality = 0.0;

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

        pos_data = PositioningData(pos_out.clone().detach());

        let norm_quality = pos_data.norm_quality().atan().clamp(0.0, 1.0) * 1.2;

        avg_norm_quality += norm_quality;


        let squeezed_class = class_out.clone().detach().squeeze_dims(&[0, 1]);

        let output_vec: Vec<f32> = class_out
            .clone()
            .detach()
            .squeeze_dims::<1>(&[0, 1])
            .to_data()
            .to_vec()
            .expect("Error unwrapping output");
        // assert_eq!(output_vec.len(), 2);

        let concentration = concentration(squeezed_class.clone());
        let can_finish = concentration > 0.5 && i >= 2;

        // println!("After argmax");

        let class_loss = mse_loss.forward(class_out, class_oh_target.clone(), Reduction::Auto);
        let class_loss_single: f32 = class_loss.clone().detach().into_data().to_vec().unwrap()[0];

        aggregate_loss += class_loss_single;

        // println!("After loss");

        let class_grad = class_loss.backward();
        let class_grad_params = GradientsParams::from_grads(class_grad, &model);

        let pos_out_dummy_diff = pos_out.mean();
        pos_out_dummy_diff_acc = Tensor::cat(vec![pos_out_dummy_diff_acc, pos_out_dummy_diff], 0);

        model = class_optim.step(model.class_lr, model, class_grad_params);

        if can_finish {
            let (highest_class, _) = tensor_argmax(squeezed_class);
            correct_output = highest_class == target;
            last_loss = class_loss.detach().to_data().to_vec().unwrap()[0];
            acc_reward -= class_loss_single;
            break;
        }

        acc_reward += class_loss_single;
    }

    avg_norm_quality /= (current_iter + 1) as f32;

    aggregate_loss /= (current_iter + 1) as f32;

    acc_reward += if correct_output { 4.0 } else { -3.0 };

    if last_loss > aggregate_loss {
        acc_reward += (last_loss - aggregate_loss) * 10.0 + 1.5;
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

    // let total_reward = acc_reward * time_goal - last_loss * 10.0 * (1.0 - time_goal) - aggregate_loss * 5.0 * (1.0 - time_goal) + 3.0;
    let total_loss = (last_loss - aggregate_loss) * 2.0 + time_needed * 0.1 + avg_norm_quality * 0.01;


    let pos_out_dummy_diff_mean = pos_out_dummy_diff_acc.mean();
    let pos_dummy_loss = pos_out_dummy_diff_mean.mul_scalar(total_loss);
    let pos_dummy_grad = pos_dummy_loss.backward();
    let pos_dummy_grad_params = GradientsParams::from_grads(pos_dummy_grad, &model);

    model = pos_optim.step(model.pos_lr, model, pos_dummy_grad_params);

    (
        model,
        StepStatistics {
            reward: -total_loss,
            last_loss,
            avg_loss: aggregate_loss,
            finished_after: current_iter + 1,
            correct: correct_output,
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
    mut data_loader: impl DataLoader<B>,
    epochs: usize,
) -> VisionModel<B> {
    let train_interrupter = TrainingInterrupter::new();
    let mut renderer = TuiMetricsRenderer::new(train_interrupter, None);

    let mut window_avg_loss_metric =
        AvgMetric::new("W Avg Loss".to_owned(), "w_avg_loss".to_owned(), 100);

    let mut window_correct_metric = AvgMetric::new(
        "W Correct guess".to_owned(),
        "w_correct_guess".to_owned(),
        100,
    );

    // println!("loading training folder");

    let mut total_step_id = 0;

    for epoch in 0..epochs {
        // println!("processing entries");
        for i in 0..data_loader.len() {
            let training_input = data_loader.next(device);

            // println!("Loading {file_name}");

            let (model_out, stats) = train(training_input, model, device, 100, optim_data);
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

            let new_window_avg_loss_metric = window_avg_loss_metric.update(stats.avg_loss as f64);
            let new_window_correct_guess_metric =
                window_correct_metric.update_saturating(if stats.correct { 100.0 } else { 0.0 });

            renderer.update_train(new_iter_metric);
            renderer.update_train(new_reward_metric);
            renderer.update_train(new_last_loss_metric);
            renderer.update_train(new_avg_loss_metric);
            renderer.update_train(new_window_avg_loss_metric);

            if let Some(w_correct) = new_window_correct_guess_metric {
                renderer.update_train(w_correct);
            }

            let training_progress = TrainingProgress {
                progress: Progress::new(data_loader.current_id(), data_loader.len()),
                epoch: epoch + 1,
                epoch_total: epochs,
                iteration: 1,
            };
            renderer.render_train(training_progress);

            // println!("{stats:#?}");

            if i % 50 == 0 {
                create_artifact_dir("model_artifacts");
                model
                    .clone()
                    .save_file("model_artifacts", &CompactRecorder::new());
            }
        }
    }

    model
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

fn concentration<B: Backend>(tensor: Tensor<B, 1>) -> f32 {
    let soft = burn::tensor::activation::softmax(tensor, 0);
    let (_, highest) = tensor_argmax(soft);
    highest
}
