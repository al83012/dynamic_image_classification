use std::{
    fs::{self, create_dir_all},
    path::{Path, PathBuf},
};

use burn::{
    module::Module,
    prelude::Backend,
    record::{self, FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
};

use crate::model::{VisionModel, VisionModelRecord};

pub fn get_highest_version(version_of: &str) -> Option<usize> {
    let folder_path_str = format!("model_artifacts/{version_of}");
    let folder_path = Path::new(&folder_path_str);
    create_dir_all(folder_path);

    let entries = fs::read_dir(folder_path)
        .expect("IO Error")
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let file_string = entry.file_name().to_string_lossy().to_string();

            let (file_name, extension) =
                file_string.rsplit_once(".").expect("Files need extensions");


            if extension.eq("mpk") { Some(()) } else { None }
                .and_then(|_| file_name.parse::<usize>().ok())
        });

    entries.max()
}

pub fn save_to_highest<B: Backend>(version_of: &str, model: &VisionModel<B>) {
    let highest = get_highest_version(version_of).unwrap_or(0);

    let file_path_str = format!("model_artifacts/{version_of}/{highest}");

    model.clone().save_file(
        file_path_str,
        &NamedMpkFileRecorder::<FullPrecisionSettings>::default(),
    );
}

pub fn load_from_highest<B: Backend>(
    version_of: &str,
    model: VisionModel<B>,
    device: &B::Device,
) -> VisionModel<B> {
    let highest = get_highest_version(version_of);

    if highest.is_none() {
        return model;
    }

    let highest = highest.unwrap();
    println!("Loading {highest}");

    let file_path_str = format!("model_artifacts/{version_of}/{highest}.mpk");
    let file_path = PathBuf::from(file_path_str);

    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
        .load(file_path.into(), device)
        .expect("Error loading model");

    model.load_record(record)
}

pub fn save_to_new_highest<B: Backend>(version_of: &str, model: &VisionModel<B>) {
    let highest = get_highest_version(version_of).map(|x| x + 1).unwrap_or(0);

    let file_path_str = format!("model_artifacts/{version_of}/{highest}");

    model.clone().save_file(
        file_path_str,
        &NamedMpkFileRecorder::<FullPrecisionSettings>::default(),
    );
}
