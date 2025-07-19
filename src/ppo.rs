use burn::{
    optim::{
        adaptor::OptimizerAdaptor, Adam, AdamState, GradientsParams, Optimizer, SimpleOptimizer,
    },
    prelude::Backend,
    tensor::backend::AutodiffBackend,
};
use image::GenericImageView;
use std::{clone::Clone, fs};

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

    let idx_tensor: Tensor<B, 1, Int> = Tensor::from_data([target], device);
    let class_oh_target_int: Tensor<B, 1, Int> = idx_tensor.one_hot(model.num_classes);
    let class_oh_target: Tensor<B, 3> = class_oh_target_int.float().unsqueeze_dims(&[0, 0]);
    let class_oh_target_vec: Vec<f32> = class_oh_target.to_data().to_vec().unwrap();

    let pos_data = PositioningData::<B>::start(device);
    let mut lstm_state: Option<LstmState<B, 2>> = None;

    let mut pos_out_dummy_diff_acc: Tensor<B, 1> = Tensor::from_data([0.0], device);

    let mse_loss = MseLoss::new();

    let mut acc_reward = 0.0;

    let mut current_iter = 0;

    let mut correct_output = false;

    let mut last_loss = 0.0;

    for i in 0..max_iter_count {
        current_iter = i;
        let ([cx, cy], rel_size) = pos_data.get_params_detach();
        let image_section = extract_section(image_tensor.clone(), cx, cy, rel_size);
        let step_in = VisionModelStepInput {
            image_section,
            pos_data: pos_data.clone(),
            lstm_state,
        };
        let step_out = model.forward(step_in);

        lstm_state = Some(step_out.next_lstm_state);

        let class_out = step_out.current_classification;
        let pos_out = step_out.next_pos.0;

        let (highest_class, certainty) = tensor_argmax(class_out.clone().squeeze_dims(&[0, 0]));

        let class_loss = mse_loss.forward(class_out, class_oh_target.clone(), Reduction::Mean);
        let mut class_reward: f32 = class_loss.clone().detach().into_data().to_vec().unwrap()[0];

        let class_grad = class_loss.backward();
        let class_grad_params = GradientsParams::from_grads(class_grad, &model);

        let pos_out_dummy_diff = pos_out.mean();
        pos_out_dummy_diff_acc = Tensor::cat(vec![pos_out_dummy_diff_acc, pos_out_dummy_diff], 0);

        model = class_optim.step(model.class_lr, model, class_grad_params);

        if certainty > 0.9 {
            if highest_class == target {
                correct_output = true;
            }
            last_loss = class_loss.detach().to_data().to_vec().unwrap()[0];
            acc_reward += class_reward;
            break;
        }

        acc_reward += class_reward;
    }

    acc_reward += if correct_output { 10.0 } else { -5.0 };

    let time_needed = (current_iter + 1) as f32 / max_iter_count as f32;

    println!("Time: {time_needed:.2}, right = {correct_output}");

    // If the answer is correct, it is better if the time is longer. If the answer is wrong, it is
    // better to see that only a short time was taken
    let time_goal = if correct_output {
        1.0 - time_needed
    } else {
        time_needed
    };

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

pub fn train_all<B: AutodiffBackend>(mut model: VisionModel<B>, device: &B::Device, optim_data: &mut OptimizerData<B>) -> VisionModel<B> {
    let training_folder =
        fs::read_dir("./data/archive/Training/Training").expect("Unable to open Training Data");
    let mut entries = training_folder.into_iter().collect::<Vec<_>>();
    let mut rng = rand::thread_rng();

    entries.shuffle(&mut rng);

    for chunk in entries.chunks(50) {
        for entry in chunk {
            let entry = entry.as_ref().expect("Unable to read dir entry");
            let file_name = entry.file_name().to_string_lossy().to_string();
            let class = if file_name.starts_with("not") { 0 } else { 1 };

            let image = load_image::<B>(&format!("Training/Training/{file_name}"), device);

            let training_input = ClassificationItem {
                image,
                classification: class,
            };

            let (model_out, stats) = train(training_input, model, device, 30, optim_data);
            model = model_out;
            println!("{stats:#?}");
        }

        todo!();
    }

    model
}
