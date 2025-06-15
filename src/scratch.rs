use burn::backend::Ndarray;
use burn::prelude::*;

type TestBackend = Ndarray;

fn main() {
    let device = burn::backend::ndarray::NdarrayDevice::default();

    let coeffs: Vec<f32> = vec![4.0, 2.0, 9.0, -5.0, 1.0];
    let len = coeffs.len();

    let coeffs_data = TensorData::new(coeffs.clone(), [len]);
    let coeffs_tensor: Tensor<TestBackend, 1> = Tensor::from_data(coeffs_data, &device);

    // Create outer product by unsqueezing and matmul
    let outer_1 = coeffs_tensor
        .clone()
        .unsqueeze_dim(1)
        .matmul(coeffs_tensor.clone().unsqueeze_dim(0));

    // Flatten the tensor
    let shape = outer_1.shape();
    let flattened = outer_1.reshape([shape.dims[0] * shape.dims[1]]);

    // Create another outer product
    let outer_2 = flattened
        .unsqueeze_dim(1)
        .matmul(coeffs_tensor.clone().unsqueeze_dim(0));

    // Reshape to cubic tensor
    let cubic_tensor = outer_2.reshape([len, len, len]);

    let val: f32 = 5.;

    let vals: Vec<f32> = vec![
        val.powi(3),
        val.powi(2),
        val.powi(1),
        val.powi(0),
        val.powi(-1),
    ];

    let powers_data = TensorData::new(vals, [5]);
    let powers: Tensor<TestBackend, 1> = Tensor::from_data(powers_data, &device);

    // Apply powers across all three dimensions
    let powers_i = powers.clone().unsqueeze_dim(1).unsqueeze_dim(2); // Shape: (5, 1, 1)
    let powers_j = powers.clone().unsqueeze_dim(0).unsqueeze_dim(2); // Shape: (1, 5, 1)
    let powers_k = powers.clone().unsqueeze_dim(0).unsqueeze_dim(1); // Shape: (1, 1, 5)

    println!("powers i shape: {:?}", powers_i.shape());
    println!("powers j shape: {:?}", powers_j.shape());
    println!("powers k shape: {:?}", powers_k.shape());

    // Element-wise multiplication across all three axes
    let result = cubic_tensor.mul(powers_i).mul(powers_j).mul(powers_k);

    // Sum all elements
    let sum = result.sum();
    let sum_data = sum.to_data();
    let sum_scalar = sum_data.as_slice::<f32>().unwrap()[0];

    println!("result summed: {}", sum_scalar);
    //output: 71414.1953
    //expected output: 205587930.8
}
