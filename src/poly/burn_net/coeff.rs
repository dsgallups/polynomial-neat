use std::fmt;

use burn::prelude::*;

use super::{basis_prime::BasisTemplate, expander::Polynomial};

#[derive(Debug)]
pub struct Coefficients<B: Backend>(Tensor<B, 2>);

impl<B: Backend> fmt::Display for Coefficients<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Coefficients({:?})", self.0.shape())
    }
}

impl<B: Backend> Coefficients<B> {
    pub fn new<T: PartialEq>(
        polynomials: &[Polynomial<T>],
        basis_template: &BasisTemplate<T>,
        device: &B::Device,
    ) -> Self {
        //so each polynomial will be represented as a row and conform to the basis template.
        //the vector will be flat and then we will reshape in the tensor.
        let mut coef_vec: Vec<f32> = vec![0.; polynomials.len() * basis_template.num_rows()];

        for (poly_i, polynomial) in polynomials.iter().enumerate() {
            for component in polynomial.components() {
                let i = basis_template
                    .position(|row| {
                        let operands = component.operands();
                        row == operands
                    })
                    .unwrap();

                let coef_vec_index = (basis_template.num_rows() * poly_i) + i;

                let val = coef_vec.get_mut(coef_vec_index).unwrap();
                *val = component.weight();
            }
        }

        println!("[");
        for (i, coef) in coef_vec.iter().enumerate() {
            if i % polynomials.len() == 0 {
                if i != 0 {
                    println!("]");
                }
                print!("[");
            }
            if coef.is_sign_positive() {
                print!(" {coef:.02},");
            } else {
                print!("{coef:.02},");
            }
            if i == coef_vec.len() - 1 {
                println!("]");
            }
        }
        println!("]");

        let data = TensorData::new(coef_vec, [polynomials.len(), basis_template.num_rows()]);
        let tensor = Tensor::from_data(data, device);

        Self(tensor)
    }

    pub fn inner(&self) -> &Tensor<B, 2> {
        &self.0
    }
}
