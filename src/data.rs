use std::{
    fs::{self, DirEntry},
    io::Error,
};

use burn::{data::dataloader::batcher::Batcher, prelude::*};
use rand::{seq::SliceRandom, thread_rng};

use crate::{image::load_image, model::VisionModel};

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

// impl<B: Backend> Batcher<B, ClassificationItem<B>, ClassificationBatch<B>>
//     for ClassificationBatcher
// {
//     fn batch(
//         &self,
//         items: Vec<ClassificationItem<B>>,
//         device: &<B as Backend>::Device,
//     ) -> ClassificationBatch<B> {
//         let images = items.iter().map(|item| item.image.clone().unsqueeze()).collect();
//         let target_one_hot_shape = [2];
//         let targets: Vec<Tensor<B, 2, Int>> = items
//             .iter()
//             .map(|item| {
//                 Tensor::from_data(TensorData::zeros::<i32, _>(target_one_hot_shape), device)
//             })
//             .collect();
//         let target_batch = Tensor::cat(targets, 0);
//         let image_batch = Tensor::cat(images, 0);
//
//         ClassificationBatch {
//             images: image_batch,
//             classifications: target_batch,
//         }
//     }
// }

pub trait DataLoader<B: Backend> {
    const NUM_CLASSES: usize;
    fn new_and_assert(model: &VisionModel<B>) -> Self;
    fn next(&mut self, device: &B::Device) -> ClassificationItem<B>;
    fn current_id(&self) -> usize;
    fn len(&self) -> usize;
}

pub struct SmokerDataLoader {
    entries: Vec<DirEntry>,
    current_entry: usize,
}

pub struct CovidDataLoader {
    entries: Vec<(String, i32)>,
    current_entry: usize,
}

impl<B: Backend> DataLoader<B> for SmokerDataLoader {
    const NUM_CLASSES: usize = 2;

    fn new_and_assert(model: &VisionModel<B>) -> Self {
        if model.num_classes != 2{
            panic!("Model class number mismatch {}!=2", model.num_classes);
        }
        let training_folder = fs::read_dir("./data/smokers/archive/Training/Training")
            .expect("Unable to open Training Data");
        let mut entries = training_folder
            .into_iter()
            .map(|entry| entry.expect("Unable to access file entry"))
            .collect::<Vec<_>>();

        let mut rng = rand::thread_rng();

        entries.shuffle(&mut rng);

        Self {
            entries,
            current_entry: 0,
        }
    }

    fn next(&mut self, device: &B::Device) -> ClassificationItem<B> {
        let binding = self.entries.get(self.current_entry);
        let next_entry = binding.as_ref().unwrap();
        self.current_entry += 1;
        if self.current_entry >= self.entries.len() {
            self.current_entry = 0;
        }

        let file_name = next_entry.file_name().to_string_lossy().to_string();

        let class = if file_name.starts_with("not") { 0 } else { 1 };

        let image = load_image::<B>(&format!("/smokers/archive/Training/Training/{file_name}"), device);

        let training_input = ClassificationItem {
            image,
            classification: class,
        };
        training_input
    }

    fn current_id(&self) -> usize {
        self.current_entry
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

impl<B: Backend> DataLoader<B> for CovidDataLoader {
    const NUM_CLASSES: usize = 3;

    fn new_and_assert(model: &VisionModel<B>) -> Self {
        
        if model.num_classes != 3{
            panic!("Model class number mismatch {}!=3", model.num_classes);
        }

        let covid_training_folder = fs::read_dir("./data/covid/Covid19-dataset/train/Covid")
            .expect("Unable to open Training Data");

        let normal_training_folder = fs::read_dir("./data/covid/Covid19-dataset/train/Normal")
            .expect("Unable to open Training Data");

        let pneumonia_training_folder = fs::read_dir("./data/covid/Covid19-dataset/train/Viral Pneumonia")
            .expect("Unable to open Training Data");

        let label_covid = covid_training_folder.into_iter().map(|entry| {
            let entry = entry.expect("Unable to access file entry");
            let file_name = entry.file_name().to_string_lossy().to_string();
            let sub_path = format!("Covid/{file_name}");

            (sub_path, 0)
        });


        let label_normal = normal_training_folder.into_iter().map(|entry| {
            let entry = entry.expect("Unable to access file entry");
            let file_name = entry.file_name().to_string_lossy().to_string();
            let sub_path = format!("Normal/{file_name}");

            (sub_path, 1)
        });


        let label_pneumonia = pneumonia_training_folder.into_iter().map(|entry| {
            let entry = entry.expect("Unable to access file entry");
            let file_name = entry.file_name().to_string_lossy().to_string();
            let sub_path = format!("Viral Pneumonia/{file_name}");

            (sub_path, 2)
        });

        let total = label_covid.chain(label_normal).chain(label_pneumonia);
        let mut total_vec = total.collect::<Vec<_>>();

        let mut rng = thread_rng();

        total_vec.shuffle(&mut rng);

        Self { entries: total_vec, current_entry: 0 }
    }

    fn next(&mut self, device: &B::Device) -> ClassificationItem<B> {
        let binding = self.entries.get(self.current_entry);
        let next_entry = binding.as_ref().unwrap();
        self.current_entry += 1;
        if self.current_entry >= self.entries.len() {
            self.current_entry = 0;
        }

        let (file_name, class) = next_entry;


        let image = load_image::<B>(&format!("/covid/Covid19-dataset/train/{file_name}"), device);

        let training_input = ClassificationItem {
            image,
            classification: *class,
        };
        training_input
    }

    fn current_id(&self) -> usize {
        self.current_entry
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}
