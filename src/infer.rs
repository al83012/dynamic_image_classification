use crate::{
    data::ClassificationItem,
    image::{extract_section, load_image},
    model::{PositioningData, VisionModel, VisionModelStepInput, VisionModelStepResult},
    train::concentration,
};
use burn::prelude::*;
use burn_train::ClassificationOutput;
use nn::LstmState;

fn single_step<B: Backend>(
    data: VisionModelStepInput<B>,
    model: &VisionModel<B>,
) -> VisionModelStepResult<B> {
    model.forward(data)
}

//image path is relative to data/ folder
pub fn steps_to_finish<B: Backend>(
    image_path: &str,
    model: &VisionModel<B>,
    device: &B::Device,
    max_iter: usize,
    concentration_limit: f32,
) -> Vec<StepInfo<B>> {
    let mut result = Vec::new();

    let mut pos_data = PositioningData::<B>::start(device);
    let mut lstm_state: Option<LstmState<B, 2>> = None;

    let image = load_image(image_path, device);

    for i in 0..max_iter {
        let time = Tensor::<B, 1>::from_data(
            TensorData::from([(i as f32 / max_iter as f32).powi(2)]),
            device,
        );
        let ([cx, cy], rel_size) = pos_data.get_params_detach();
        let image_section = extract_section(image.clone(), cx, cy, rel_size);
        let step_in = VisionModelStepInput {
            image_section,
            pos_data: pos_data.clone(),
            lstm_state,
            time
        };

        let step_out = model.forward(step_in);

        let step_info = StepInfo {
            pos_in: pos_data.clone(),
            pos_out: step_out.next_pos.clone(),
            class_out: step_out.current_classification.clone(),
        };

        result.push(step_info);

        lstm_state = Some(step_out.next_lstm_state);

        let class_out = step_out.current_classification;
        let pos_out = step_out.next_pos.0;

        pos_data = PositioningData(pos_out.clone().detach());

        let squeezed_class = class_out.clone().detach().squeeze_dims(&[0, 1]);

        let output_vec: Vec<f32> = class_out
            .clone()
            .detach()
            .squeeze_dims::<1>(&[0, 1])
            .to_data()
            .to_vec()
            .expect("Error unwrapping output");

        let concentration = concentration(squeezed_class.clone());
        let can_finish = concentration > concentration_limit && i >= 1;
    }

    result
}

pub struct StepInfo<B: Backend> {
    pub pos_in: PositioningData<B>,
    pub pos_out: PositioningData<B>,
    pub class_out: Tensor<B, 3>,
}
