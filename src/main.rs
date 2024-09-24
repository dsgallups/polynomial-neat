use candle_core::{Device, Result, Tensor};
/*
fn main() -> Result<()> {
    let coeffs: Vec<f32> = vec![4.0, 2.0, 9.0, -5.0, 1.0];
    let val: f32 = 5.0;

    let vals: [f32; 5] = [
        val.powi(3),
        val.powi(2),
        val.powi(1),
        val.powi(0),
        val.powi(-1),
    ];
    let powers = Tensor::new(&vals, &Device::Cpu)?;

    let coeffs_tensor = Tensor::new(coeffs, &Device::Cpu)?;
    let product = coeffs_tensor.mul(&powers)?;

    let sum = product.sum_all()?; // This is sum(coeffs[i] * powers[i})
    println!("sum: {}", sum);
    //let result = sum;

    Ok(())
}*/

fn main() -> Result<()> {
    let coeffs: Vec<f32> = vec![4.0, 2.0, 9.0, -5.0, 1.0];
    let len = coeffs.len();

    let coeffs_tensor = Tensor::new(coeffs, &Device::Cpu)?;

    let outer_1 = coeffs_tensor
        .unsqueeze(1)?
        .matmul(&coeffs_tensor.unsqueeze(0)?)?;

    let flattened = outer_1.flatten(0, 1)?;

    let outer_2 = flattened
        .unsqueeze(1)?
        .matmul(&coeffs_tensor.unsqueeze(0)?)?;

    let cubic_tensor = outer_2.reshape((len, len, len))?;

    let val: f32 = 5.;

    let vals: [f32; 5] = [
        val.powi(3),
        val.powi(2),
        val.powi(1),
        val.powi(0),
        val.powi(-1),
    ];

    let powers = Tensor::new(&vals, &Device::Cpu)?;

    // Apply powers across all three dimensions
    let powers_i = powers.unsqueeze(1)?.unsqueeze(2)?; // Shape: (5, 1, 1)
    let powers_j = powers.unsqueeze(0)?.unsqueeze(2)?; // Shape: (1, 5, 1)
    let powers_k = powers.unsqueeze(0)?.unsqueeze(1)?; // Shape: (1, 1, 5)

    println!(
        "powers i, j, k:\n{}\n{}\n{}\n",
        powers_i, powers_j, powers_k
    );

    //let result = cubic_tensor.mul(&powers.expand((5, 5, 5))?)?;
    // Element-wise multiplication across all three axes
    let result = cubic_tensor
        .mul(&powers_i)?
        .mul(&powers_j)?
        .mul(&powers_k)?;

    println!("result summed: {}", result.sum_all()?);
    //output: 71414.1953
    //expected output: 205587930.8

    Ok(())
}
