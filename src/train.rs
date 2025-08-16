use burn::backend::autodiff::grads::Gradients;
use burn::config::Config;
use burn::data::dataloader::Progress;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::loss;
use burn_train::metric::MetricEntry;
use burn_train::renderer::tui::TuiMetricsRenderer;
use burn_train::renderer::{self, MetricState, MetricsRenderer, TrainingProgress};
use burn_train::TrainingInterrupter;
use log::Level;
use nannou::conrod_core::render;
use nn::loss::{MseLoss, Reduction};
use nn::LstmState;

use crate::data::{ClassificationItem, DataLoader};
use crate::image::extract_section;
use crate::metrics::AvgMetric;
use crate::model::{PositioningData, VisionModel, VisionModelStepInput};
use crate::pos_opt::PosOptimizationStrategy;
use crate::save;

#[derive(Debug, Clone, Copy)]
pub struct StepStatistics {
    pub reward: f32,
    pub first_loss: f32,
    pub last_loss: f32,
    pub avg_loss: f32,
    pub finished_after: usize,
    pub target: usize,
    pub last_out: usize,
    pub avg_step_improvement: f32,
}

pub struct OptimizerData<B: AutodiffBackend> {
    pub class_optim: OptimizerAdaptor<Adam, VisionModel<B>, B>,
    pub pos_optim: OptimizerAdaptor<Adam, VisionModel<B>, B>,
}

#[derive(Config)]
pub struct TrainingConfig {
    #[config(default = "20")]
    max_iter_count: usize,
    #[config(default = "40")]
    epochs: usize,
    save_as: String,
    #[config(default = "0.012")]
    norm_quality_weight: f32,
    #[config(default = "(0.4, 0.9)")]
    certainty_slope: (f32, f32),
    #[config(default = "2.0")]
    iter_improvement_weight: f32,
    #[config(default = "0.2")]
    iter_time_weight: f32,
    #[config(default = "1.5e-5")]
    class_lr: f64,
    #[config(default = "5e-5")]
    pos_lr: f64,
    #[config(default = "0.8")]
    lr_min_decayed_portion: f64,
}

pub struct TrainingManager<B: AutodiffBackend> {
    optimizers: OptimizerData<B>,
    config: TrainingConfig,
    device: B::Device,
    gradient_accum: GradientsAccumulator<VisionModel<B>>,
    pos_opt: PosOptimizationStrategy<B>,
}

impl<B: AutodiffBackend> TrainingManager<B> {
    pub fn init(config: TrainingConfig, device: B::Device) -> Self {
        let optimizers = OptimizerData {
            class_optim: AdamConfig::new().init(),
            pos_optim: AdamConfig::new().init(),
        };

        let pos_opt = PosOptimizationStrategy::new(config.max_iter_count, device.clone());

        Self {
            optimizers,
            config,
            device,
            gradient_accum: GradientsAccumulator::new(),
            pos_opt,
        }
    }
    fn train(
        &mut self,
        data: ClassificationItem<B>,
        model: VisionModel<B>,
        t: f32,
    ) -> (VisionModel<B>, StepStatistics) {
        let lr_portion = ((1.0 - t) + self.config.lr_min_decayed_portion as f32 * t) as f64;
        let adj_class_lr = lr_portion * self.config.class_lr;
        let adj_pos_lr = lr_portion * self.config.pos_lr;
        let mut model = model;

        let class_optim = &mut self.optimizers.class_optim;
        let pos_optim = &mut self.optimizers.pos_optim;

        let image_tensor = data.image;
        let target = data.classification as usize;

        // let idx_tensor: Tensor<B, 1, Int> = Tensor::from_data([target], device);
        // let class_oh_target_int: Tensor<B, 1, Int> = idx_tensor.one_hot(model.num_classes);
        // let class_oh_target: Tensor<B, 3> = class_oh_target_int.float().unsqueeze_dims(&[0, 0]);
        // let class_oh_target_vec: Vec<f32> = class_oh_target.to_data().to_vec().unwrap();

        let mut class_oh_vec = vec![0.0; model.num_classes];
        class_oh_vec[target] = 1.0;
        let class_oh_target: Tensor<B, 3> =
            Tensor::<B, 1>::from_data(class_oh_vec.as_slice(), &self.device)
                .unsqueeze_dims(&[0, 1]);

        self.pos_opt.new_step(class_oh_target.clone());

        // println!("Extracted class data");

        let mut pos_data = PositioningData::<B>::start(&self.device);
        let mut lstm_state: Option<Vec<LstmState<B, 2>>> = None;

        // let mut pos_out_dummy_diff_acc: Tensor<B, 1> = Tensor::from_data([0.0], &self.device);

        let mse_loss = MseLoss::new();

        // let mut acc_reward = 0.0;

        let mut current_iter = 0;

        let mut last_guess = 0;

        let mut last_loss = 0.0;

        let mut aggregate_loss = 0.0;

        let mut avg_norm_quality = 0.0;

        let mut first_loss = 0.0;

        #[cfg(not(feature = "no_class_proc"))]
        let mut class_grad_accum = GradientsAccumulator::new();

        // #[cfg(not(feature = "no_pos_proc"))]
        // let mut pos_grad_accum = GradientsAccumulator::new();

        // let mut previous_loss: Tensor<B, 1> =
        //     Tensor::from_data(TensorData::from([0.0]), &self.device);

        // let mut class_improvement_grad_accum = GradientsAccumulator::new();

        let mut aggregate_loss_improvement: f32 = 0.0;

        // println!("Setup for iteration");
        // log::info!("Setup for iteration");

        // log::info!("New item");

        for i in 0..self.config.max_iter_count {
            // log::info!("Step: {i}");
            let time_val = i as f32 / self.config.max_iter_count as f32;
            let time =
                Tensor::<B, 1>::from_data(TensorData::from([time_val.powi(2)]), &self.device);
            // log::info!("Iteration Start [{i}]");
            current_iter = i;
            // println!("Iter[{current_iter:?}]");
            let ([cx, cy], rel_size) = pos_data.get_params_detach();
            // log::info!("After pos_data_unpacking");
            let image_section = extract_section(image_tensor.clone(), cx, cy, rel_size);
            let step_in = VisionModelStepInput {
                image_section,
                pos_data: pos_data.clone(),
                lstm_state,
                time,
            };

            // log::info!("Loaded input");

            // println!("Before fwd");
            let step_out = model.forward(step_in);

            // log::info!("Did forward");

            // println!("After fwd");
            lstm_state = Some(step_out.next_lstm_state);

            let class_out = softmax( step_out.current_classification, 2);
            let new_pos_data = step_out.next_pos;
            let pos_out = new_pos_data.clone().0;
            #[cfg(feature = "no_pos_proc")]
            {
                pos_data = PositioningData::random(&self.device);
            }
            #[cfg(not(feature = "no_pos_proc"))]
            {
                pos_data = PositioningData(pos_out.clone().detach());
            }

            // log::info!("After repackaging pos data");

            let norm_quality = pos_data.norm_quality().atan().clamp(0.0, 1.0);

            avg_norm_quality += norm_quality;

            #[cfg(not(feature = "no_pos_proc"))]
            {
                self.pos_opt
                    .accumulate_substep(class_out.clone(), new_pos_data);
            }

            // let output_vec: Vec<f32> = class_out
            //     .clone()
            //     .detach()
            //     .squeeze_dims::<1>(&[0, 1])
            //     .to_data()
            //     .to_vec()
            //     .expect("Error unwrapping output");

            // assert_eq!(output_vec.len(), 2);

            let squeezed_class = class_out.clone().detach().squeeze_dims(&[0, 1]);

            let concentration = concentration(squeezed_class.clone());
            // println!("Concentration: {concentration:.2}");
            let concentration_limit =
                self.config.certainty_slope.0 * (1.0 - t) + self.config.certainty_slope.1 * (t);
            let can_finish = concentration > concentration_limit && i >= 1;

            // log::info!("After concentration calc");

            // println!("After argmax");

            let class_adj_strength = smoothstep(time_val * 2.0);
            // println!("Target: {class_adj_strength:.2}");
            let class_adj_target = class_oh_target.clone() * class_adj_strength;

            // log::info!("Adj target: {class_adj_target:.2}");

            let class_loss = mse_loss.forward(class_out.clone(), class_adj_target, Reduction::Auto);
            let class_loss_full =
                mse_loss.forward(class_out, class_oh_target.clone(), Reduction::Auto);
            let class_loss_single: f32 = class_loss_full
                .clone()
                .detach()
                .into_data()
                .to_vec()
                .unwrap()[0];

            if i == 0 {
                first_loss = class_loss_single;
            }

            aggregate_loss += class_loss_single;

            // println!("After loss");

            #[cfg(not(feature = "no_class_proc"))]
            {
                let class_grad = class_loss.clone().backward();
                let class_grad_params = GradientsParams::from_grads(class_grad, &model);
                class_grad_accum.accumulate(&model, class_grad_params);
            }

            // log::info!("did loss and gradients");

            // NOTE: Recently deactivated
            // let pos_out_dummy_diff = pos_out.clone().mean();
            // pos_out_dummy_diff_acc =
            //     Tensor::cat(vec![pos_out_dummy_diff_acc, pos_out_dummy_diff], 0);

            if i % 5 == 0 {
                #[cfg(not(feature = "no_class_proc"))]
                {
                    model = class_optim.step(adj_class_lr, model, class_grad_accum.grads());
                }
            }

            if i > 1 {
                // The lower, the better the model is performing
                // Or rather: negative is good
                //
                // NOTE: Recently deactivated
                // let loss_change = class_loss_full.clone() - previous_loss.clone();
                // previous_loss = class_loss_full.clone();
                // let loss_change_f = loss_change.to_data().to_vec::<f32>().unwrap()[0];
                // aggregate_loss_improvement -= loss_change_f;

                // let pos_grad = pos_out.mul_scalar(-loss_change).backward();
                // let pos_grad_params = GradientsParams::from_grads(pos_grad, &model);
                // pos_grad_accum.accumulate(&model, pos_grad_params);

                // let grads = GradientsParams::from_grads((-loss_change).backward(), &model);

                // class_improvement_grad_accum.accumulate(&model, grads);

                // self.gradient_accum.accumulate(&model, grads);
            }

            // log::info!("Concentration: {concentration:.2}");
            // log::info!("PredClass: {squeezed_class:.2}");

            if (can_finish && i + 1 >= 2) || i + 1 == self.config.max_iter_count {
                let (highest_class, _) = tensor_argmax(squeezed_class);
                last_guess = highest_class;
                last_loss = //class_loss.detach().to_data().to_vec().unwrap()[0];
                    class_loss_single;
                // acc_reward -= class_loss_single;
                break;
            }

            // acc_reward += class_loss_single;
        }

        //NOTE: This needs to be reactivated later

        #[cfg(not(feature = "no_class_proc"))]
        {
            model = class_optim.step(adj_class_lr, model, class_grad_accum.grads());
        }

        // log::info!("Passed iterations");

        avg_norm_quality /= (current_iter + 1) as f32;

        aggregate_loss /= (current_iter + 1) as f32;

        let avg_loss_improvement = aggregate_loss_improvement / (current_iter as f32).max(1.0);

        // acc_reward += if correct_output { 4.0 } else { -3.0 };

        // if last_loss > aggregate_loss {
        //     acc_reward += (last_loss - aggregate_loss) * 10.0 + 1.5;
        // }

        let time_needed = (current_iter + 1) as f32 / self.config.max_iter_count as f32;

        // println!("Time: {time_needed:.2}, right = {correct_output}");

        // If the answer is correct, it is better if the time is longer. If the answer is wrong, it is
        // better to see that only a short time was taken
        // let time_goal = if correct_output {
        //     1.0 - time_needed
        // } else {
        //     time_needed
        // };

        // let total_reward = acc_reward * time_goal - last_loss * 10.0 * (1.0 - time_goal) - aggregate_loss * 5.0 * (1.0 - time_goal) + 3.0;

        // let avg_improvement_loss = aggregate_loss_improvement / (current_iter  + 1) as f32;

        let total_loss = (last_loss - aggregate_loss) * self.config.iter_improvement_weight
            + time_needed * time_needed * self.config.iter_time_weight
            + avg_norm_quality * self.config.norm_quality_weight;

        // let total_loss = last_loss
        //     + (last_loss - aggregate_loss) * self.config.iter_improvement_weight
        //     + time_needed * self.config.iter_time_weight + 10.0;

        // let total_loss = if last_loss > aggregate_loss {
        //     if last_loss < 0.1 {
        //         0.0
        //     } else {
        //         -1.0
        //     }
        // } else {
        //     1.0
        // };
        // - 1.0
        // + time_needed * time_needed * self.config.iter_time_weight
        // + avg_norm_quality * self.config.norm_quality_weight;

        #[cfg(not(feature = "no_pos_proc"))]
        {
            let grads = self.pos_opt.apply(&model);
            model = pos_optim.step(adj_pos_lr, model, grads);
        }

        // NOTE: Recently deactivated
        // let pos_out_dummy_diff_mean = pos_out_dummy_diff_acc.mean();
        // let pos_dummy_loss = pos_out_dummy_diff_mean.mul_scalar(-total_loss);
        // let pos_dummy_grad = pos_dummy_loss.backward();

        // #[cfg(not(feature = "no_pos_proc"))]
        // let gradients = pos_grad_accum.grads();

        // NOTE: Recently deactivated
        // let pos_dummy_grad_params = GradientsParams::from_grads(pos_dummy_grad, &model);
        // self.gradient_accum
        //     .accumulate(&model, pos_dummy_grad_params);

        // model = pos_optim.step(adj_pos_lr, model, pos_dummy_grad_params * total_loss);

        // model = pos_optim.step(adj_pos_lr, model, pos_grad_accum.grads());

        // model = pos_optim.step(adj_pos_lr, model, class_improvement_grad_accum.grads());

        // log::info!("Did full train for image");
        (
            model,
            StepStatistics {
                first_loss,
                reward: -total_loss,
                last_loss,
                avg_loss: aggregate_loss,
                finished_after: current_iter + 1,
                last_out: last_guess,
                target,
                avg_step_improvement: avg_loss_improvement,
            },
        )
    }

    pub fn train_all(
        &mut self,
        mut model: VisionModel<B>,
        mut data_loader: impl DataLoader<B>,
    ) -> VisionModel<B> {
        let train_interrupter = TrainingInterrupter::new();
        let mut renderer = TuiMetricsRenderer::new(train_interrupter, None);

        let mut window_avg_loss_metric =
            AvgMetric::new("W Avg Loss".to_owned(), "w_avg_loss".to_owned(), 250);

        let mut window_correct_metric =
            AvgMetric::new("W %".to_owned(), "w_correct_guess".to_owned(), 250);

        let mut window_correct_covid_metric = AvgMetric::new(
            "W % COVID".to_owned(),
            "w_correct_guess_covid".to_owned(),
            100,
        );

        let mut window_correct_normal_metric = AvgMetric::new(
            "W % NORMAL".to_owned(),
            "w_correct_guess_normal".to_owned(),
            100,
        );

        let mut window_correct_pneumonia_metric = AvgMetric::new(
            "W % PNEUM".to_owned(),
            "w_correct_guess_pneumonia".to_owned(),
            100,
        );
        let mut window_avg_iter =
            AvgMetric::new("W Iter".to_owned(), "w_iter_number".to_owned(), 250);

        let mut window_avg_loss_improvement =
            AvgMetric::new("D_Loss / iter".to_owned(), "d_loss_iter".to_owned(), 250);

        let mut window_first_last_improvement = AvgMetric::new(
            "W F/L".to_owned(),
            "w_first_last_improvement".to_owned(),
            250,
        );

        save::save_to_new_highest(&self.config.save_as, &model);

        // println!("loading training folder");

        for epoch in 0..self.config.epochs {
            // println!("processing entries");
            let t = epoch as f32 / self.config.epochs as f32;
            for i in 0..data_loader.len() {
                let training_input = data_loader.next(&self.device);

                // println!("Loading {file_name}");
                // log::info!("Before step (file = {i}, epoch = {i})");
                let (model_out, stats) = self.train(training_input, model, t);

                // log::info!("After step (file = {i}, epoch = {epoch})");
                model = model_out;

                let new_iter_metric = MetricState::Numeric(
                    MetricEntry {
                        name: "Iter".to_string(),
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
                        name: "Last L".to_string(),
                        formatted: "Last Loss".to_string(),
                        serialize: "last_loss".to_string(),
                    },
                    stats.last_loss as f64,
                );

                let new_avg_loss_metric = MetricState::Numeric(
                    MetricEntry {
                        name: "Avg L".to_string(),
                        formatted: "Avg Loss".to_string(),
                        serialize: "avg_loss".to_string(),
                    },
                    stats.avg_loss as f64,
                );

                let new_first_last_metric = MetricState::Numeric(
                    MetricEntry {
                        name: "F/L".to_string(),
                        formatted: "F/L".to_string(),
                        serialize: "first_last_improvement".to_string(),
                    },
                    stats.first_loss as f64 - stats.last_loss as f64,
                );

                let new_window_avg_loss_metric =
                    window_avg_loss_metric.update(stats.avg_loss as f64);
                let new_window_correct_guess_metric =
                    window_correct_metric.update_saturating(if stats.last_out == stats.target {
                        100.0
                    } else {
                        0.0
                    });

                if let Some(w_iter_num) =
                    window_avg_iter.update_saturating(stats.finished_after as f64)
                {
                    renderer.update_train(w_iter_num);
                }

                let next_cov_stat = if stats.target == 0 {
                    Some(if stats.last_out == 0 { 100.0 } else { 0.0 })
                } else {
                    None
                };
                let next_norm_stat = if stats.target == 1 {
                    Some(if stats.last_out == 1 { 100.0 } else { 0.0 })
                } else {
                    None
                };
                let next_pneum_stat = if stats.target == 2 {
                    Some(if stats.last_out == 2 { 100.0 } else { 0.0 })
                } else {
                    None
                };

                if let Some(cov_m) =
                    window_correct_covid_metric.update_saturating_or_avg(next_cov_stat)
                {
                    renderer.update_train(cov_m);
                }

                if let Some(norm_m) =
                    window_correct_normal_metric.update_saturating_or_avg(next_norm_stat)
                {
                    renderer.update_train(norm_m);
                }

                if let Some(pneum_m) =
                    window_correct_pneumonia_metric.update_saturating_or_avg(next_pneum_stat)
                {
                    renderer.update_train(pneum_m);
                }

                if let Some(improvement) =
                    window_avg_loss_improvement.update_saturating(stats.avg_step_improvement as f64)
                {
                    renderer.update_train(improvement);
                }

                if let Some(fl_imp) = window_first_last_improvement
                    .update_saturating(stats.first_loss as f64 - stats.last_loss as f64)
                {
                    renderer.update_train(fl_imp);
                }

                renderer.update_train(new_iter_metric);
                renderer.update_train(new_reward_metric);
                renderer.update_train(new_last_loss_metric);
                renderer.update_train(new_avg_loss_metric);
                renderer.update_train(new_window_avg_loss_metric);
                renderer.update_train(new_first_last_metric);

                if let Some(w_correct) = new_window_correct_guess_metric {
                    renderer.update_train(w_correct);
                }

                let training_progress = TrainingProgress {
                    progress: Progress::new(data_loader.current_id(), data_loader.len()),
                    epoch: epoch + 1,
                    epoch_total: self.config.epochs,
                    iteration: 1,
                };
                renderer.render_train(training_progress);

                // println!("{stats:#?}");

                // if i % 50 == 0 {
                //     create_artifact_dir("model_artifacts");
                //     model
                //         .clone()
                //         .save_file("model_artifacts", &CompactRecorder::new());
                // }

                //NOTE: OG
                #[cfg(not(feature = "no_pos_proc"))]
                {
                    if i % 5 == 0 {
                        let lr_portion =
                            ((1.0 - t) + self.config.lr_min_decayed_portion as f32 * t) as f64;
                        let adj_pos_lr = self.config.pos_lr * lr_portion;
                        model = self.optimizers.pos_optim.step(
                            adj_pos_lr,
                            model,
                            self.gradient_accum.grads(),
                        );
                    }
                }
            }

            if epoch % 5 == 0 {
                save::save_to_highest(&self.config.save_as, &model);
            }
        }

        model
    }
}

pub fn concentration<B: Backend>(tensor: Tensor<B, 1>) -> f32 {
    let soft = burn::tensor::activation::softmax(tensor, 0);
    let (_, highest) = tensor_argmax(soft);
    highest
}

pub fn tensor_argmax<B: Backend>(t: Tensor<B, 1>) -> (usize, f32) {
    let vec = t.into_data().to_vec::<f32>().unwrap();
    vec.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(a, b)| (a, *b))
        .unwrap()
}

fn smoothstep(val: f32) -> f32 {
    let val = val.clamp(0.0, 1.0);
    val * val * (3.0 - 2.0 * val)
}
