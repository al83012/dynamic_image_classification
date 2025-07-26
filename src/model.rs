use burn::{
    optim::{Adam, AdamConfig},
    prelude::*,
};
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
    linear_pos: Linear<B>,
    linear_class: Linear<B>,
    activation: Relu,
    pub class_lr: f64,
    pub pos_lr: f64,
    pub num_classes: usize,
}

#[derive(Config)]
pub struct VisionModelConfig {
    num_classes: usize,
    #[config(default = "128")]
    lstm_output_size: usize,
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
    #[config(default = "[10, 10]")]
    pool_out: [usize; 2],
    #[config(default = "0.1")]
    dropout: f64,
    #[config(default = "5e-5")]
    class_lr: f64,
    #[config(default = "7e-5")]
    pos_lr: f64,
}

impl VisionModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VisionModel<B> {
        let positioning_data_size = PositioningData::<B>::SIZE;
        let conv_1_out_channel = self.color_channels * self.conv_1_kernel_count;
        let conv_2_out_channel = conv_1_out_channel * self.conv_2_kernel_count;
        
        let conv_1_out_img_dim = [self.pool_out[0] - 2, self.pool_out[1] - 2];
        let conv_2_out_img_dim = [conv_1_out_img_dim[0] - 2, conv_1_out_img_dim[1] - 2];

        let conv_2_out_flat = conv_2_out_channel * conv_2_out_img_dim[0] * conv_2_out_img_dim[1];
        let lstm_input_size = conv_2_out_flat + positioning_data_size;
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
            lstm: LstmConfig::new(lstm_input_size, self.lstm_hidden_size, false).init(device),
            linear_pos: LinearConfig::new(self.lstm_hidden_size, positioning_data_size)
                .init(device),
            linear_class: LinearConfig::new(self.lstm_output_size, self.num_classes).init(device),
            activation: Relu::new(),
            num_classes: self.num_classes,
            class_lr: self.class_lr,
            pos_lr: self.pos_lr,
        }
    }
}

#[derive(Clone)]
pub struct PositioningData<B: Backend>(pub Tensor<B, 3>);

pub struct VisionModelStepResult<B: Backend> {
    pub current_classification: Tensor<B, 3>,
    pub next_pos: PositioningData<B>,
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
        let cat_vec = vec![x, squeezed_pos_data];
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

impl<B: Backend> PositioningData<B> {
    pub const SIZE: usize = 3;
    pub fn from_params(
        section_center: [f32; 2],
        selection_coverage: f32,
        device: &B::Device,
    ) -> Self {
        Self(
            Tensor::<B, 1>::from_floats(
                [section_center[0], section_center[1], selection_coverage],
                device,
            )
            .unsqueeze_dims(&[0, 0]),
        )
    }
    pub fn start(device: &B::Device) -> Self {
        Self::from_params([0.0, 0.0], 1.0, device)
    }
    pub fn get_params_detach(&self) -> ([f32; 2], f32) {
        let data = self.0.clone().detach().to_data();
        let vec: Vec<f32> = data
            .to_vec()
            .expect("PositioningData should be able to be converted to Vec");
        assert!(vec.len() == 3);
        ([vec[0], vec[1]], vec[2])
    }

    pub fn norm_quality(&self) -> f32 {
        let ([cx, cy], size) = self.get_params_detach();
        let cx_norm = (cx.abs() - 2.0).max(0.0);
        let cy_norm = (cy.abs() - 2.0).max(0.0);
        let size_norm = (size.abs() - 1.0).max(0.0) + if size.is_sign_negative() {1.0} else {0.0};
        cx_norm + cy_norm + size_norm
    }
}
