use burn::prelude::*;
use nn::LstmState;
use rand::Rng;


#[derive(Clone)]
pub struct PositioningData<B: Backend>(pub Tensor<B, 3>);

pub struct VisionModelStepResult<B: Backend> {
    pub current_classification: Tensor<B, 3>,
    pub next_pos: PositioningData<B>,
    pub next_lstm_state: Vec<LstmState<B, 2>>,
}

pub struct VisionModelStepInput<B: Backend> {
    pub image_section: Tensor<B, 3>, // [Channels, Width, Height]
    pub pos_data: PositioningData<B>,
    pub lstm_state: Option<Vec<LstmState<B, 2>>>,
    pub time: Tensor<B, 1>,
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
        let size_norm =
            (size.abs() - 1.0).max(0.0) + if size.is_sign_negative() { 1.0 } else { 0.0 };
        cx_norm + cy_norm + size_norm
    }
    pub fn random(device: &B::Device) -> Self {
        let mut rng = rand::rng();
        let cx = rng.random_range(-1.0..1.0) * 4.0;
        let cy = rng.random_range(-1.0..1.0) * 4.0;
        let size = rng.random_range(0.1..0.9);

        Self::from_params([cx, cy], size, device)
    }
}
