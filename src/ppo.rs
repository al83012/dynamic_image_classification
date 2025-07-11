use burn::{
    optim::{adaptor::OptimizerAdaptor, Adam, AdamState, GradientsParams, Optimizer, SimpleOptimizer},
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


pub struct OptimizerData<B: AutodiffBackend> {
    class_optim: OptimizerAdaptor<Adam, VisionModel<B>, B>,   
    pos_optim: OptimizerAdaptor<Adam, VisionModel<B>, B>,   
}

pub fn train<B: AutodiffBackend>(
    data: ClassificationItem<B>,
    model: VisionModel<B>,
    device: &B::Device,
    max_iter_count: usize,
    optim_data: &mut OptimizerData<B>,
) -> VisionModel<B> {

    let mut model = model;

    let class_optim = &mut optim_data.class_optim;
    let pos_optim = &mut optim_data.pos_optim;


    let image_tensor = data.image;
    let target = data.classification as usize;

    let idx_tensor: Tensor<B, 1, Int> = Tensor::from_data([target], device);
    let class_oh_target_int: Tensor<B, 1, Int> = idx_tensor.one_hot(model.num_classes);
    let class_oh_target: Tensor<B, 3> = class_oh_target_int.float().unsqueeze_dims(&[0, 0]);
    let class_oh_target_vec: Vec<f32> = class_oh_target.to_data().to_vec().unwrap();

    let mut pos_data = PositioningData::<B>::start(device);
    let mut lstm_state: Option<LstmState<B, 2>> = None;

    let mse_loss = MseLoss::new();
    let mut total_classification_loss: Tensor<B, 1> = Tensor::from_data([0.0], device);

    let mut aggregate_reward = 0.0;

    let mut current_iter = 0;

    for i in 0..max_iter_count {
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

        let class_loss = mse_loss.forward(class_out, class_oh_target.clone(), Reduction::Mean);
        let class_grad = class_loss.backward();
        let class_grad_params = GradientsParams::from_grads(class_grad, &model);
        model = class_optim.step(model.class_lr, model, class_grad_params);


        todo!("Still need to accumulate the gradients for the positioning")
    }



    todo!("Still need to apply the loss for positioning")

}

fn tensor_argmax<B: Backend>(t: Tensor<B, 1>) -> (usize, f32) {
    let vec = t.into_data().to_vec::<f32>().unwrap();
    vec.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(a, b)| (a, *b))
        .unwrap()
}
