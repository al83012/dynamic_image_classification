use burn::prelude::*;
use nn::{
    conv::{Conv2d, Conv2dConfig},
    pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    Dropout, Linear, LinearConfig, Lstm, LstmConfig, LstmState, Relu,
};

#[derive(Module, Debug)]
pub struct VisionModel<B: Backend> {
    pool: AdaptiveAvgPool2d,
    conv_1: Conv2d<B>,
    conv_2: Conv2d<B>,
    dropout: Dropout,
    lstm: Lstm<B>,
    linear_2: Linear<B>,
    activation: Relu,
    pub num_classes: usize,
}

#[derive(Config, Debug)]
pub struct VisionModelConfig {
    num_classes: usize,
    #[config(default = "128")]
    linear_hidden_size: usize,
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
    #[config(default = "8")]
    conv_2_kernel_count: usize,
    #[config(default = "[8, 8]")]
    pool_out: [usize; 2],
    #[config(default = "0.4")]
    dropout: f64,
}

impl VisionModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VisionModel<B> {
        let positioning_data_size = PositioningData::<B>::SIZE;
        let conv_1_out_channel = self.color_channels * self.conv_1_kernel_count;
        let conv_2_out_channel = conv_1_out_channel * self.conv_2_kernel_count;
        // Pool out is the size every image is scaled down to
        let conv_2_out_flat = conv_2_out_channel * self.pool_out[0] * self.pool_out[1];
        let lstm_input_size = conv_2_out_flat + positioning_data_size;
        let model_output_size = self.num_classes + positioning_data_size;
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
            lstm: LstmConfig::new(lstm_input_size, self.lstm_hidden_size, false).init(device),
            linear_2: LinearConfig::new(self.linear_hidden_size, model_output_size).init(device),
            activation: Relu::new(),
            num_classes: self.num_classes,
        }
    }
}

#[derive( Clone)]
pub struct PositioningData<B: Backend>(pub Tensor<B, 1>);

pub struct VisionModelStepResult<B: Backend> {
    pub current_classification: Tensor<B, 1>,
    pub next_pos_data: PositioningData<B>,
    pub next_lstm_state: LstmState<B, 2>,
}

pub struct VisionModelStepInput<B: Backend> {
    pub image_section: Tensor<B, 3>, // [Channels, Width, Height]
    pub pos_data: PositioningData<B>,
    pub lstm_state: Option<LstmState<B, 2>>,
}

impl<B: Backend> VisionModel<B> {
    pub fn forward(&self, input: VisionModelStepInput<B>) -> VisionModelStepResult<B> {
        let image_section = input.image_section;
        let pos_data = input.pos_data;

        let image_batched = image_section.unsqueeze();

        let x = self.pool.forward(image_batched);
        let x = self.conv_1.forward(x);
        let x = self.conv_2.forward(x);
        //Flatten the image-dim
        let x: Tensor<B, 2> = x.flatten(1, 3); //[batch_size,
                                               //flattened_image]
        let unsqueezed_pos_data = pos_data.0.unsqueeze(); // [batch_size, pos_data]

        // Concat the flattened image and pos data (while keeping the 1-sized sequence and batch)
        let cat_vec = unsafe { vec![x, unsqueezed_pos_data] };
        let x = Tensor::cat(cat_vec, 2).unsqueeze();
        let (x_out, next_state) = self.lstm.forward(x, input.lstm_state);
        let x = self.linear_2.forward(x_out);
        // Remove unused dims batchsize and seq_len
        let x: Tensor<B, 1> = x.unsqueeze_dims(&[0, 1]);
        let mut split = x
            .split_with_sizes(vec![self.num_classes, PositioningData::<B>::SIZE], 0)
            .into_iter();
        let classification = split.next().unwrap();
        let next_pos_data = split.next().unwrap();

        VisionModelStepResult {
            current_classification: classification,
            next_pos_data: PositioningData::from_tensor(next_pos_data.detach()), 
            next_lstm_state: next_state,
        }
    }

    pub fn infer(&self, image: Tensor<B, 3>) -> Tensor<B, 1> {
        todo!("Do the iterations...")
    }
}

impl<B: Backend> PositioningData<B> {
    pub const SIZE: usize = 3;
    pub fn from_params(
        section_center: [f32; 2],
        selection_coverage: f32,
        device: &B::Device,
    ) -> Self {
        Self(Tensor::from_floats(
            [section_center[0], section_center[1], selection_coverage],
            device,
        ))
    }
    pub fn from_tensor(tensor: Tensor<B, 1>) -> Self {
        Self(tensor)
    }
    pub fn start(device: &B::Device) -> Self {
        Self::from_params([0.5, 0.5], 1.0, device)
    }
    pub fn get_params(
        &self
    ) -> ([f32;2], f32) {
        let data = self.0.to_data();
        let vec: Vec<f32> = data.to_vec().expect("PositioningData should be able to be converted to Vec");
        assert!(vec.len() == 3);
        ([vec[0], vec[1]], vec[2])
    }
}
