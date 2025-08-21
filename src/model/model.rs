use burn::{
    optim::{Adam, AdamConfig},
    prelude::*,
};
use nn::{
    conv::{Conv2d, Conv2dConfig},
    pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    Dropout, Linear, LinearConfig, Lstm, LstmConfig, LstmState, Relu,
};
use rand::Rng;

use crate::model::modern_lstm::{StackedLstm, StackedLstmConfig};

use super::{PositioningData, VisionModelStepInput, VisionModelStepResult};

#[derive(Module, Debug)]
pub struct VisionModel<B: Backend> {
    pool: AdaptiveAvgPool2d,
    conv_1: Conv2d<B>,
    conv_2: Conv2d<B>,
    dropout: Dropout,
    lstm: StackedLstm<B>,
    linear_pos: Linear<B>,
    linear_class: Linear<B>,
    activation: Relu,
    pub num_classes: usize,
}

#[derive(Config)]
pub struct VisionModelConfig {
    num_classes: usize,
    #[config(default = "128")]
    lstm_hidden_size: usize,
    #[config(default = "[3, 3]")]
    conv_1_kernel: [usize; 2],
    #[config(default = "[3, 3]")]
    conv_2_kernel: [usize; 2],
    #[config(default = "3")]
    color_channels: usize,
    #[config(default = "8")]
    conv_1_kernel_count: usize,
    #[config(default = "16")]
    conv_2_kernel_count: usize,
    #[config(default = "[8, 8]")]
    pool_out: [usize; 2],
    #[config(default = "0.2")]
    dropout: f64,
}

impl VisionModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VisionModel<B> {
        let positioning_data_size = PositioningData::<B>::SIZE;
        let conv_1_out_channel = self.color_channels * self.conv_1_kernel_count;
        let conv_2_out_channel = conv_1_out_channel * self.conv_2_kernel_count;

        let conv_1_out_img_dim = [self.pool_out[0] - 2, self.pool_out[1] - 2];
        let conv_2_out_img_dim = [conv_1_out_img_dim[0] - 2, conv_1_out_img_dim[1] - 2];

        let conv_2_out_flat = conv_2_out_channel * conv_2_out_img_dim[0] * conv_2_out_img_dim[1];
        let time_size = 1;
        let lstm_input_size = conv_2_out_flat + positioning_data_size + time_size;
        let model_total_output_size = self.num_classes + positioning_data_size;

        // println!("conv_2_out_channel = {}", conv_2_out_channel);
        // println!("conv_2_out_flat = {}", conv_2_out_flat);
        // println!("positioning_data_size = {}", positioning_data_size);
        // println!("lstm_input_size = {}", lstm_input_size);

        VisionModel {
            pool: AdaptiveAvgPool2dConfig::new(self.pool_out).init(),
            conv_1: Conv2dConfig::new(
                [self.color_channels, conv_1_out_channel],
                self.conv_1_kernel,
            )
            .init(device),
            conv_2: Conv2dConfig::new([conv_1_out_channel, conv_2_out_channel], self.conv_2_kernel)
                .init(device),
            dropout: Dropout { prob: self.dropout },
            lstm: StackedLstmConfig::new(lstm_input_size, self.lstm_hidden_size, 2, 0.1)
                .init(device),
            linear_pos: LinearConfig::new(self.lstm_hidden_size, positioning_data_size)
                .init(device),
            linear_class: LinearConfig::new(self.lstm_hidden_size, self.num_classes).init(device),
            activation: Relu::new(),
            num_classes: self.num_classes,
        }
    }
}

impl<B: Backend> VisionModel<B> {
    pub fn forward(&self, input: VisionModelStepInput<B>) -> VisionModelStepResult<B> {
        let image_section = input.image_section;
        let pos_data = input.pos_data;

        let image_batched = image_section.unsqueeze();

        let x = self.pool.forward(image_batched);

        // let pool_out_shape = x.shape();

        // println!("Pool output shape: {pool_out_shape:?}");
        let x = self.conv_1.forward(x);

        // let conv_1_out_shape = x.shape();

        // println!("Conv 1 output shape: {conv_1_out_shape:?}");
        let x = self.conv_2.forward(x);

        let x = self.dropout.forward(x);

        // let conv_2_out_shape = x.shape();
        // println!("Conv 2 output shape: {conv_2_out_shape:?}");
        //Flatten the image-dim
        let x: Tensor<B, 2> = x.flatten(1, 3); //[batch_size, image net out]
                                               //flattened_image]
        let squeezed_pos_data = pos_data.0.squeeze(0); // [batch_size, pos_data]

        // Concat the flattened image and pos data (while keeping the 1-sized sequence and batch)
        let cat_vec = vec![x, squeezed_pos_data, input.time.unsqueeze()];
        let x = Tensor::cat(cat_vec, 1).unsqueeze();
        // println!("Before Lstm");
        let (x_out, next_state) = self.lstm.forward(x, input.lstm_state);

        // println!("Before Linear class");
        let classification = self.linear_class.forward(x_out.clone());

        // println!("Before Linear pos");
        let next_pos_data = self.linear_pos.forward(x_out);

        VisionModelStepResult {
            current_classification: classification,
            next_pos: PositioningData(next_pos_data),
            next_lstm_state: next_state,
        }
    }
}
