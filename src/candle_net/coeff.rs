use candle_core::{Device, Result, Tensor};

use super::{basis_prime::BasisTemplate, expander::Polynomial};

pub struct Coefficients(Tensor);

impl Coefficients {
    pub fn new<T: PartialEq>(
        polynomials: &[Polynomial<T>],
        basis_template: &BasisTemplate<T>,
        device: &Device,
    ) -> Result<Self> {
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

        let tensor = Tensor::new(coef_vec, device)?
            .reshape((polynomials.len(), basis_template.num_rows()))?;

        Ok(Self(tensor))
    }
    pub fn inner(&self) -> &Tensor {
        &self.0
    }
}
