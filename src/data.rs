use burn::{data::dataloader::batcher::Batcher, prelude::*};

pub struct ClassificationBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub classifications: Tensor<B, 2, Int>,
}

pub struct ClassificationItem<B: Backend> {
    pub image: Tensor<B, 3>,
    pub classification: i32,
}

pub struct ClassificationBatcher {
    pub num_classes: usize,
}

impl<B: Backend> Batcher<B, ClassificationItem<B>, ClassificationBatch<B>>
    for ClassificationBatcher
{
    fn batch(
        &self,
        items: Vec<ClassificationItem<B>>,
        device: &<B as Backend>::Device,
    ) -> ClassificationBatch<B> {
        let images = items.iter().map(|item| item.image.clone().unsqueeze()).collect();
        let target_one_hot_shape = [2];
        let targets: Vec<Tensor<B, 2, Int>> = items
            .iter()
            .map(|item| {
                Tensor::from_data(TensorData::zeros::<i32, _>(target_one_hot_shape), device)
            })
            .collect();
        let target_batch = Tensor::cat(targets, 0);
        let image_batch = Tensor::cat(images, 0);

        ClassificationBatch {
            images: image_batch,
            classifications: target_batch,
        }
    }
}


