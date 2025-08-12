// use burn::{
//     data::dataloader::Progress,
//     optim::{
//         adaptor::OptimizerAdaptor, Adam, AdamState, GradientsParams, Optimizer, SimpleOptimizer,
//     },
//     prelude::Backend,
//     record::CompactRecorder,
//     tensor::backend::AutodiffBackend,
// };
// use image::GenericImageView;
// use std::{clone::Clone, fs};
//
// use burn_train::{
//     renderer::{tui::TuiMetricsRenderer, MetricState, MetricsRenderer, TrainingProgress},
//     SummaryMetrics, TrainingInterrupter,
// };
//
// use rand::seq::SliceRandom;
// use rand::thread_rng;
//
// use burn::prelude::*;
// use nn::{
//     loss::{MseLoss, Reduction},
//     LstmState,
// };
//
// use crate::{
//     data::{ClassificationItem, DataLoader},
//     image::{extract_section, load_image},
//     metrics::AvgMetric,
//     model::{PositioningData, VisionModel, VisionModelStepInput},
// };
//
//
//
// #[derive(Debug, Clone, Copy)]
// pub struct StepStatistics {
//     pub reward: f32,
//     pub last_loss: f32,
//     pub avg_loss: f32,
//     pub finished_after: usize,
//     pub correct: bool,
// }
//
// pub fn train<B: AutodiffBackend>(
//     data: ClassificationItem<B>,
//     model: VisionModel<B>,
//     device: &B::Device,
//     max_iter_count: usize,
//     optim_data: &mut OptimizerData<B>,
//     t: f32,
// ) -> (VisionModel<B>, StepStatistics) {
//     let mut model = model;
//
//     let class_optim = &mut optim_data.class_optim;
//     let pos_optim = &mut optim_data.pos_optim;
//
//     let image_tensor = data.image;
//     let target = data.classification as usize;
//
//     // let idx_tensor: Tensor<B, 1, Int> = Tensor::from_data([target], device);
//     // let class_oh_target_int: Tensor<B, 1, Int> = idx_tensor.one_hot(model.num_classes);
//     // let class_oh_target: Tensor<B, 3> = class_oh_target_int.float().unsqueeze_dims(&[0, 0]);
//     // let class_oh_target_vec: Vec<f32> = class_oh_target.to_data().to_vec().unwrap();
//
//     let mut class_oh_vec = vec![0.0; model.num_classes];
//     class_oh_vec[target] = 1.0;
//     let class_oh_target: Tensor<B, 3> =
//         Tensor::<B, 1>::from_data(class_oh_vec.as_slice(), device).unsqueeze_dims(&[0, 1]);
//
//     // println!("Extracted class data");
//
//     let mut pos_data = PositioningData::<B>::start(device);
//     let mut lstm_state: Option<LstmState<B, 2>> = None;
//
//     let mut pos_out_dummy_diff_acc: Tensor<B, 1> = Tensor::from_data([0.0], device);
//
//     let mse_loss = MseLoss::new();
//
//     let mut acc_reward = 0.0;
//
//     let mut current_iter = 0;
//
//     let mut correct_output = false;
//
//     let mut last_loss = 0.0;
//
//     let mut aggregate_loss = 0.0;
//
//     let mut avg_norm_quality = 0.0;
//
//     // println!("Setup for iteration");
//     // log::info!("Setup for iteration");
//
//     for i in 0..max_iter_count {
//         // log::info!("Iteration Start [{i}]");
//         current_iter = i;
//         // println!("Iter[{current_iter:?}]");
//         let ([cx, cy], rel_size) = pos_data.get_params_detach();
//         // log::info!("After pos_data_unpacking");
//         let image_section = extract_section(image_tensor.clone(), cx, cy, rel_size);
//         let step_in = VisionModelStepInput {
//             image_section,
//             pos_data: pos_data.clone(),
//             lstm_state,
//         };
//
//         // log::info!("Loaded input");
//
//         // println!("Before fwd");
//         let step_out = model.forward(step_in);
//
//         // log::info!("Did forward");
//
//         // println!("After fwd");
//         lstm_state = Some(step_out.next_lstm_state);
//
//         let class_out = step_out.current_classification;
//         let pos_out = step_out.next_pos.0;
//
//         pos_data = PositioningData(pos_out.clone().detach());
//
//         // log::info!("After repackaging pos data");
//
//         let norm_quality = pos_data.norm_quality().atan().clamp(0.0, 1.0);
//
//         avg_norm_quality += norm_quality;
//
//         let squeezed_class = class_out.clone().detach().squeeze_dims(&[0, 1]);
//
//         let output_vec: Vec<f32> = class_out
//             .clone()
//             .detach()
//             .squeeze_dims::<1>(&[0, 1])
//             .to_data()
//             .to_vec()
//             .expect("Error unwrapping output");
//
//         // assert_eq!(output_vec.len(), 2);
//
//         let concentration = concentration(squeezed_class.clone());
//         let concentration_limit = 0.4 * (1.0 - t) + 0.6 * (t);
//         let can_finish = concentration > concentration_limit && i >= 1;
//
//         // log::info!("After concentration calc");
//
//         // println!("After argmax");
//
//         let class_loss = mse_loss.forward(class_out, class_oh_target.clone(), Reduction::Auto);
//         let class_loss_single: f32 = class_loss.clone().detach().into_data().to_vec().unwrap()[0];
//
//         aggregate_loss += class_loss_single;
//
//         // println!("After loss");
//
//         let class_grad = class_loss.backward();
//         let class_grad_params = GradientsParams::from_grads(class_grad, &model);
//
//         // log::info!("did loss and gradients");
//
//         let pos_out_dummy_diff = pos_out.mean();
//         pos_out_dummy_diff_acc = Tensor::cat(vec![pos_out_dummy_diff_acc, pos_out_dummy_diff], 0);
//
//         model = class_optim.step(self.class_lr, model, class_grad_params);
//
//         if can_finish {
//             let (highest_class, _) = tensor_argmax(squeezed_class);
//             correct_output = highest_class == target;
//             last_loss = class_loss.detach().to_data().to_vec().unwrap()[0];
//             acc_reward -= class_loss_single;
//             break;
//         }
//
//         acc_reward += class_loss_single;
//     }
//
//     // log::info!("Passed iterations");
//
//     avg_norm_quality /= (current_iter + 1) as f32;
//
//     aggregate_loss /= (current_iter + 1) as f32;
//
//     acc_reward += if correct_output { 4.0 } else { -3.0 };
//
//     if last_loss > aggregate_loss {
//         acc_reward += (last_loss - aggregate_loss) * 10.0 + 1.5;
//     }
//
//     let time_needed = (current_iter + 1) as f32 / max_iter_count as f32;
//
//     // println!("Time: {time_needed:.2}, right = {correct_output}");
//
//     // If the answer is correct, it is better if the time is longer. If the answer is wrong, it is
//     // better to see that only a short time was taken
//     // let time_goal = if correct_output {
//     //     1.0 - time_needed
//     // } else {
//     //     time_needed
//     // };
//
//     // let total_reward = acc_reward * time_goal - last_loss * 10.0 * (1.0 - time_goal) - aggregate_loss * 5.0 * (1.0 - time_goal) + 3.0;
//     let total_loss =
//         (last_loss - aggregate_loss) * 2.0 + time_needed * 0.1 + avg_norm_quality * 0.012;
//
//     let pos_out_dummy_diff_mean = pos_out_dummy_diff_acc.mean();
//     let pos_dummy_loss = pos_out_dummy_diff_mean.mul_scalar(total_loss);
//     let pos_dummy_grad = pos_dummy_loss.backward();
//     let pos_dummy_grad_params = GradientsParams::from_grads(pos_dummy_grad, &model);
//
//     model = pos_optim.step(self.pos_lr, model, pos_dummy_grad_params);
//
//     // log::info!("Did full train for image");
//     (
//         model,
//         StepStatistics {
//             reward: -total_loss,
//             last_loss,
//             avg_loss: aggregate_loss,
//             finished_after: current_iter + 1,
//             correct: correct_output,
//         },
//     )
// }
//
//
//
//
//
//
