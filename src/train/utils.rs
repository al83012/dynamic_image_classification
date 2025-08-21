use burn::prelude::*;


pub fn concentration<B: Backend>(tensor: Tensor<B, 1>) -> f32 {
    let soft = burn::tensor::activation::softmax(tensor, 0);
    let (_, highest) = tensor_argmax(soft);
    highest
}

pub fn tensor_argmax<B: Backend>(t: Tensor<B, 1>) -> (usize, f32) {
    let vec = t.into_data().to_vec::<f32>().unwrap();
    vec.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(a, b)| (a, *b))
        .unwrap()
}

pub fn smoothstep(val: f32) -> f32 {
    let val = val.clamp(0.0, 1.0);
    val * val * (3.0 - 2.0 * val)
}
