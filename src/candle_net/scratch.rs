use std::marker::PhantomData;

use super::{expander::Polynomial, CandleNetwork};
use crate::{
    candle_net::{
        basis_prime::BasisTemplate, coeff::Coefficients, expander::Variable,
        get_topology_polynomials,
    },
    prelude::*,
};
use candle_core::{Device, Result, Tensor};
use fnv::FnvHashMap;
use pretty_assertions::assert_eq;
use uuid::Uuid;

/*
//  now we need to figure how to create B' for the provided inputs
//  we will have [1, input1, input2, ...] as the input, multiplied by this vector to get our
//  basis matrix.
// if we have 2 inputs, and there are 5 values in the basis matrix, the template will be 5x3.
let num_cols = inputs.len() + 1;
let num_rows = basis.len();
//let basis_prime_template: [[usize; num_cols]; num_rows] = [[0; num_cols]; num_rows];
let basis_prime_template: BasisPrimeTemplate<usize> =
    BasisPrimeTemplate::new(num_cols, num_rows);*/
#[test]
fn scratch() -> Result<()> {
    let x_id = Uuid::new_v4();
    let y_id = Uuid::new_v4();

    println!("Input 1 id: {}\nInput 2 id: {}", x_id, y_id);

    let x_n = arc(NeuronTopology::input(x_id));
    let y_n = arc(NeuronTopology::input(y_id));

    let hidden_one = arc(NeuronTopology::hidden(
        Uuid::new_v4(),
        vec![
            InputTopology::downgrade(&x_n, 3., 1),
            InputTopology::downgrade(&y_n, 1., 1),
        ],
    ));

    // (3x + y )^2 =
    // 9x^2 + 6xy + y^2
    let output_1 = arc(NeuronTopology::output(
        Uuid::new_v4(),
        vec![InputTopology::downgrade(&hidden_one, 1., 2)],
    ));

    // 2(3x + y)
    //
    // 6x + 2y
    let output_2 = arc(NeuronTopology::output(
        Uuid::new_v4(),
        vec![InputTopology::downgrade(&hidden_one, 2., 1)],
    ));

    let topology = NetworkTopology::from_raw_parts(
        vec![x_n, y_n, hidden_one, output_1, output_2],
        MutationChances::none(),
    );
    let inputs: FnvHashMap<Uuid, usize> = topology
        .neuron_ids()
        .into_iter()
        .enumerate()
        .map(|(v, k)| (k, v))
        .collect();
    let mut output_polynomials = get_topology_polynomials(&topology)
        .into_iter()
        .map(|poly| poly.map_operands(&inputs))
        .collect::<Vec<_>>();

    output_polynomials
        .iter_mut()
        .for_each(|poly| poly.sort_by_exponent(0));

    println!("converted:\n{:#?}", output_polynomials);

    /*
       x^2
       xy
       y^2
       x
       y
    */
    let basis = super::basis_prime::basis_from_poly_list(&output_polynomials);

    println!("basis:\n{:#?}", basis);
    assert_eq!(basis[0], &[Variable::new(1, 2)]); //y^2
                                                  /* Commented
                                                  assert_eq!(basis[1], &[Variable::new(0, 1), Variable::new(1, 1)]);
                                                  xy
                                                  */
    assert_eq!(basis[2], &[Variable::new(0, 2)]); //x^2
    assert_eq!(basis[3], &[Variable::new(1, 1)]); //x
    assert_eq!(basis[4], &[Variable::new(0, 1)]); //y

    let basis_template: BasisTemplate<usize> = BasisTemplate::from_raw(basis);

    let predictions: [(usize, f32); 2] = [(0, 3.0), (1, 2.0)];

    /*
       if x is 3 and y is 2, the following should be true

       Basis matrix:
       [
       [4],
       [6],
       [9],
       [2],
       [3]
       ]
    */

    let basis_tensor = basis_template.make_tensor(predictions, &Device::Cpu)?;
    // now make the tensor for our coefficients
    let coef = Coefficients::new(&output_polynomials, &basis_template, &Device::Cpu);

    println!("basis tensor: {}", basis_tensor);

    Ok(())
}

#[test]
fn test_tensor() {
    let my_tensor = Tensor::new(vec![1., 2., 3.], &candle_core::Device::Cpu).unwrap();

    println!("tensor: {}", my_tensor.unsqueeze(1).unwrap());

    // really a vec![vec![5; 2]; 3];
    let tensor_two_vec = vec![5.; 6];
    let my_tensor_two = Tensor::new(tensor_two_vec, &Device::Cpu).unwrap();
    println!("tensor_two: \n{}\n", my_tensor_two);
    let my_tensor_two = my_tensor_two.reshape((2, 3)).unwrap();
    println!("tensor_two reshaped: \n{}\n", my_tensor_two);
}
#[test]
fn old_main() -> Result<()> {
    let coeffs: Vec<f32> = vec![4.0, 2.0, 9.0, -5.0, 1.0];
    let len = coeffs.len();

    let coeffs_tensor = Tensor::new(coeffs, &Device::Cpu)?;

    let outer_1 = coeffs_tensor.unsqueeze(1)?;

    println!("outer_1 unsqueezed: {:?}", outer_1);

    let outer_1 = outer_1.matmul(&coeffs_tensor.unsqueeze(0)?)?;

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
