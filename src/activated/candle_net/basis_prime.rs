/*pub struct BasisPrimeTemplate<T>(Vec<Vec<T>>);

impl<T: Default + Clone> BasisPrimeTemplate<T> {
    // creates a zeroed prime template with the default value of T
    pub fn new(num_cols: usize, num_rows: usize) -> Self {
        Self(vec![vec![T::default(); num_cols]; num_rows])
    }
}

#[derive(Clone, Copy, Default)]
pub enum TemplateValue<T> {
    Zero,
    One,

}*/

use candle_core::{Device, Result, Tensor};
use fnv::FnvHashMap;
use std::env::var;
use std::hash::Hash;

use super::expander::{Polynomial, Variable};

/// a single column matrix
#[derive(Debug)]
pub struct BasisTemplate<T>(Vec<Vec<Variable<T>>>);

impl<T> BasisTemplate<T> {
    /// The tensor generated will ALWAYS have one column.
    pub fn num_rows(&self) -> usize {
        self.0.len()
    }
    pub fn from_raw(basis_vec: Vec<Vec<Variable<T>>>) -> Self {
        Self(basis_vec)
    }

    pub fn position<F>(&self, predicate: F) -> Option<usize>
    where
        F: FnMut(&Vec<Variable<T>>) -> bool,
    {
        self.0.iter().position(predicate)
    }

    pub fn get(&self, index: usize) -> Option<&[Variable<T>]> {
        self.0.get(index).map(|v| v.as_slice())
    }

    pub fn new(polynomials: &[Polynomial<T>]) -> Self
    where
        T: Clone + PartialEq,
    {
        let basis_vec = basis_from_poly_list(polynomials);
        Self::from_raw(basis_vec)
    }
    pub fn make_tensor(
        &self,
        variables: impl IntoIterator<Item = (T, f32)>,
        device: &Device,
    ) -> Result<Tensor>
    where
        T: Hash + Eq + Clone,
    {
        let hashmap: FnvHashMap<T, f32> = variables.into_iter().collect();
        let mut values: Vec<f32> = Vec::with_capacity(self.0.len());

        for template_vars in self.0.iter() {
            let mut running_val = 1.;
            for template_var in template_vars {
                let Some(input_val) = hashmap.get(template_var.var()) else {
                    panic!("input val not found");
                };

                running_val *= (*input_val).powi(template_var.exponent());
            }

            values.push(running_val);
        }
        //now make a tensor
        Tensor::new(values, device)?.unsqueeze(1)
    }
}

/// returns a basis that will be used to calculate two other matrices, to be explained
pub(super) fn basis_from_poly_list<T: Clone + PartialEq>(
    polynomials: &[Polynomial<T>],
) -> Vec<Vec<Variable<T>>> {
    //the first thing we need to do is determine what our basis matrix looks like.
    // this is the set of used combinations for all polynomials.
    let mut used_combinations: Vec<Vec<Variable<T>>> = Vec::new();

    for polynomial in polynomials {
        for component in polynomial.components() {
            if !used_combinations.contains(&component.operands) {
                used_combinations.push(component.operands.clone());
            }
        }
    }

    used_combinations
}
