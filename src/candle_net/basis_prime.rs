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

use candle_core::Tensor;

use super::expander::{Polynomial, Variable};

/// a single column matrix
pub struct BasisTemplate<T>(Vec<Vec<Variable<T>>>);

impl<T: PartialEq + Clone> BasisTemplate<T> {
    pub fn new(polynomials: Vec<Polynomial<T>>) -> Self {
        let basis_vec = basis_from_poly_list(&polynomials);
        Self::from_raw(basis_vec)
    }

    pub fn from_raw(basis_vec: Vec<Vec<Variable<T>>>) -> Self {
        Self(basis_vec)
    }

    pub fn make_tensor(variables: &[T]) -> Tensor {
        todo!();
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
