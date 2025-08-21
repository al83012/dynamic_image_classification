use std::collections::VecDeque;

use burn::optim::{GradientsAccumulator, GradientsParams};
use burn::prelude::*;
use burn::{
    nn::loss::CrossEntropyLoss,
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Int, Shape, Tensor},
};

use crate::model::{PositioningData, VisionModel};

use super::utils::tensor_argmax;

pub struct PosOptimizationStrategy<B: AutodiffBackend> {
    pub total_items: usize,
    pub item_counts: [usize; 3],
    pub guesses: VecDeque<usize>,
    pub max_iter: usize,
    pub current_iter: usize,
    pub target: usize,
    pub dummy_pos_outs: Vec<(Tensor<B, 3>, f32)>,
    pub prev_loss: f32,
    pub acc_loss_reduction: f32,
    pub cross_entropy: CrossEntropyLoss<B>,
    pub device: B::Device,
}

impl<B: AutodiffBackend> PosOptimizationStrategy<B> {
    pub fn new(max_iter: usize, device: B::Device) -> Self {
        let cross_entropy = burn::nn::loss::CrossEntropyLossConfig::new().init(&device);

        Self {
            total_items: 0,
            item_counts: [0, 0, 0],
            guesses: VecDeque::with_capacity(250),
            max_iter,
            current_iter: 0,
            target: 0,
            dummy_pos_outs: Vec::new(),
            cross_entropy,
            acc_loss_reduction: 0.0,
            prev_loss: 0.0,
            device,
        }
    }
    pub fn accumulate_substep(&mut self, class_out: Tensor<B, 3>, pos_out: PositioningData<B>) {
        self.current_iter += 1;

        let squeezed_class_out: Tensor<B, 2> = class_out.detach().squeeze(0);
        let class_full_loss = self
            .cross_entropy
            .forward(squeezed_class_out.clone(), self.target_tensor());
        let full_loss_value: f32 = class_full_loss.to_data().to_vec().unwrap()[0];

        let loss_reduction = if self.current_iter > 1 {
            self.prev_loss - full_loss_value
        } else {
            0.0
        };
        self.acc_loss_reduction += loss_reduction;
        self.prev_loss = full_loss_value;

        let pos_quality = pos_out.norm_quality();

        let (guess_index, _) = tensor_argmax(squeezed_class_out.squeeze(0));
        let distribution_eq_loss = self.distribution_eq_loss(guess_index);

        let full_loss =
            self.prev_loss - loss_reduction * 3.0 + pos_quality * 0.1 + distribution_eq_loss * 3.0;

        let pos_out_tensor = pos_out.0;
        self.dummy_pos_outs.push((pos_out_tensor, full_loss));
    }

    fn target_tensor(&self) -> Tensor<B, 1, Int> {
        Tensor::from_data(TensorData::from([self.target]), &self.device)
    }

    pub fn new_step(&mut self, target: Tensor<B, 3>) {
        let (target_id, _) = tensor_argmax(target.squeeze_dims(&[0, 1]));
        self.prev_loss = 0.0;
        self.current_iter = 0;
        self.target = target_id;
        self.acc_loss_reduction = 0.0;
        self.dummy_pos_outs = vec![];
    }

    fn distribution_eq_loss(&mut self, guess_index: usize) -> f32 {
        let window_len = self.guesses.len();
        self.item_counts[guess_index] += 1;
        self.guesses.push_back(guess_index);
        if window_len + 1 > 250 {
            let remove_from_count = self.guesses.pop_front().unwrap();
            self.item_counts[remove_from_count] -= 1;
        }

        if guess_index == self.target {
            // We won't penalize correct guesses, no matter how unbalanced they are
            return 0.0;
        }

        self.square_normalization()[guess_index]
    }

    fn square_normalization(&self) -> [f32; 3] {
        let squares = self.item_counts.iter().map(|x| (*x as f32).powi(2));
        let square_sum: f32 = squares.clone().sum();
        let mut normalized = squares.map(|x| x / square_sum);

        [
            normalized.next().unwrap(),
            normalized.next().unwrap(),
            normalized.next().unwrap(),
        ]
    }

    pub fn apply(&mut self, model: &VisionModel<B>) -> GradientsParams {
        let time_needed = self.current_iter as f32 / self.max_iter as f32;
        let time_factor = time_needed * time_needed * 7.0;

        let mut gradient_accum = GradientsAccumulator::new();

        let mut base_tensors = Vec::new();

        std::mem::swap(&mut base_tensors, &mut self.dummy_pos_outs);

        base_tensors
            .into_iter()
            .map(|(tensor, local_loss)| tensor.mul_scalar(local_loss + time_factor))
            .for_each(|tensor| {
                let grad = tensor.backward();
                let grad_params = GradientsParams::from_grads(grad, model);
                gradient_accum.accumulate(model, grad_params);
            });

        gradient_accum.grads()
    }
}
