use burn::{
    optim::{Optimizer, SimpleOptimizer},
    prelude::Backend,
    tensor::backend::AutodiffBackend,
};
use image::GenericImageView;
use std::clone::Clone;

use burn::prelude::*;
use nn::{
    loss::{MseLoss, Reduction},
    LstmState,
};

use crate::{
    data::ClassificationItem,
    image::extract_section,
    model::{PositioningData, VisionModel, VisionModelStepInput},
};

pub fn train<B: Backend>(
    data: ClassificationItem<B>,
    model: &VisionModel<B>,
    device: &B::Device,
    optimizer: impl SimpleOptimizer<B>,
    max_iter_count: usize,
) {
    let image = data.image;
    let target = data.classification as usize;

    let idx_tensor: Tensor<B, 1, Int> = Tensor::from_data([target], device);
    let one_hot_int = idx_tensor.one_hot(model.num_classes);
    let one_hot_float: Tensor<B, 1> = one_hot_int.float();
    let one_hot_target_vec: Vec<f32> = one_hot_float.to_data().to_vec().unwrap();

    let mut pos_data = PositioningData::<B>::start(device);
    let mut lstm_state: Option<LstmState<B, 2>> = None;

    let mse_loss = MseLoss::new();
    let mut total_classification_loss = 0.0;

    let mut aggregate_reward = 0.0;

    let mut current_iter = 0;

    for i in 0..max_iter_count {
        current_iter = i;
        let ([cx, cy], size) = pos_data.get_params();
        let image_section = extract_section(image.clone(), cx, cy, size);

        let step_input = VisionModelStepInput::<B> {
            image_section,
            pos_data: pos_data.clone(),
            lstm_state,
        };

        let out = model.forward(step_input);

        pos_data = out.next_pos_data.clone();
        lstm_state = Some(out.next_lstm_state);

        let class_pred_tensor = out.current_classification.detach();

        let class_pred_vec: Vec<f32> = class_pred_tensor
            .to_data()
            .to_vec()
            .expect("Should be possible to put to vec");

        let sq_error: f32 = class_pred_vec
            .iter()
            .zip(&one_hot_target_vec)
            .map(|(a, b)| (*a - *b).powi(2))
            .sum();

        let classification_loss: f32 = mse_loss.forward(
            class_pred_tensor.clone(),
            one_hot_float.clone(),
            Reduction::Auto,
        ).to_data().to_vec().unwrap()[0];

        total_classification_loss = total_classification_loss + classification_loss;

        let (highest_idx, highest_val) = tensor_argmax(class_pred_tensor);

        let mut reward = (1.0 / sq_error.max(0.01));

        if highest_idx == target {
            reward += if highest_val > 0.9 { 1.0 } else { 5.0 };
        }
        aggregate_reward += reward;
        if highest_val > 0.9 {
            break;
        }
    }

    let avg_reward = aggregate_reward / (current_iter + 1) as f32;
    let avg_class_loss = total_classification_loss / (current_iter + 1) as f32;

    println!("AVG CLASS LOSS: {avg_class_loss}");

    let iter_termination_reward = avg_reward  * (1.0 -(((current_iter + 1) as f32) / max_iter_count as f32).sqrt());

    let full_loss = 1.0 / iter_termination_reward.sqrt();
}

fn tensor_argmax<B: Backend>(t: Tensor<B, 1>) -> (usize, f32) {
    let vec = t.into_data().to_vec::<f32>().unwrap();
    vec.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(a, b)| (a, *b))
        .unwrap()
}
