use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt::Debug,
    hash::{BuildHasher, Hash},
    ops::{Mul, MulAssign},
};
mod indeterminant;
use candle_core::{Device, Tensor};
pub use indeterminant::*;
mod polycomponent;
pub use polycomponent::*;
use uuid::Uuid;

/*

#[doc = r#"

n = number of additive operations
v = number of considered variables
"#]
#[derive(Debug, Clone)]
pub struct Polynomial<'a> {
    device: &'a Device,
    // 1xv matrix
    // type u32.
    var_order: Tensor,
    // nx1 matrix, type f32
    coef: Tensor,
    // nx1xv matrix, type i32
    exponents: Tensor,
}
*/

#[derive(Debug, Clone)]
pub struct Polynomial<'a> {
    device: &'a Device,
    ops: Vec<PolyComponent>,
}

impl<'a> Polynomial<'a> {
    pub fn parts(&self) -> &[PolyComponent] {
        &self.ops
    }

    pub fn components(&self) -> &[PolyComponent] {
        &self.ops
    }
    pub fn into_components(self) -> Vec<PolyComponent> {
        self.ops
    }
}

impl<'dev> Polynomial<'dev> {
    pub fn new(device: &'dev Device) -> Self {
        Self {
            device,
            ops: Vec::new(),
        }
    }
    pub fn add_operation<'other, I: Indeterminate<'other>>(
        self,
        weight: f32,
        indeterminant: I,
        exponent: i32,
    ) -> Self
    where
        'dev: 'other,
    {
        let result = indeterminant.apply_operation(self.device, weight, exponent);
        todo!()
    }

    pub fn with_device<'new>(self, device: &'new Device) -> Polynomial<'new> {
        Polynomial {
            device,
            ops: self.ops,
        }
    }

    /*pub fn add_indeterminate<'other, I: Indeterminate<'other, Variable = T>>(
        mut self,
        indeterminant: I,
    ) -> Self {
        todo!()
    }*/

    pub fn from_polycomponent(device: &'dev Device, component: PolyComponent) -> Self {
        Self {
            device,
            ops: vec![component],
        }
    }
    pub fn unit(device: &'dev Device, var: usize) -> Self {
        Self {
            device,
            ops: vec![PolyComponent::simple(1., var, 1)],
        }
    }

    /// Returns a unique list of symbolic variables
    pub fn variadics(&self) -> BTreeSet<usize> {
        self.ops.iter().fold(BTreeSet::new(), |mut acc, op| {
            for var in op.operands().keys() {
                acc.insert(*var);
            }
            acc
        })
    }
}

impl<'a> Polynomial<'a> {
    pub fn with_capacity(device: &'a Device, cap: usize) -> Self {
        Self {
            device,
            ops: Vec::with_capacity(cap),
        }
    }
    pub fn with_operation(mut self, weight: f32, variable: usize, exponent: i32) -> Self {
        self.handle_operation(weight, variable, exponent);
        self
    }

    pub fn with_polycomponent(mut self, component: PolyComponent) -> Self {
        self.handle_polycomponent(component);
        self
    }

    pub fn handle_operation(&mut self, weight: f32, variable: usize, exponent: i32) -> &mut Self {
        self.handle_polycomponent(PolyComponent::simple(weight, variable, exponent))
    }
    pub fn handle_polycomponent(&mut self, component: PolyComponent) -> &mut Self {
        match self
            .ops
            .iter_mut()
            .find(|op| op.operands == component.operands)
        {
            Some(op) => {
                op.adjust_weight(component.weight());
            }
            None => self.ops.push(component),
        }
        self
    }

    /// raises the whole polynomial to the power of -1.
    ///
    /// In turn, all of the exponents are multiplied by -1.
    pub fn invert(&mut self) {
        for component in self.ops.iter_mut() {
            for (_, exp) in component.operands.iter_mut() {
                *exp *= -1;
            }
        }
    }

    /// FOIL
    fn mul_expand<'b>(self, other: &'b Polynomial<'b>) -> Polynomial<'a> {
        let mut result = Polynomial::with_capacity(
            self.device,
            self.components().len().max(other.components().len()) * 2,
        ); // a guesstimate

        for c1 in self.into_components() {
            for c2 in other.components() {
                let together = c1.clone() * c2.clone();
                result.handle_polycomponent(together);
            }
        }

        result
    }

    pub fn expand(&mut self, other: Polynomial, weight: f32, exponent: i32) -> &mut Self {
        // important to clone here since mutating other will multiply the exponents.

        if exponent == 0 {
            self.handle_polycomponent(PolyComponent::base(weight));
            return self;
        }

        let mut running = other.clone();

        for _ in 1..exponent.abs() {
            running = running.mul_expand(&other);
        }

        if exponent < 0 {
            running.invert();
        }

        running *= weight;

        for component in running.into_components() {
            self.handle_polycomponent(component);
        }

        self
    }
}

impl<'a> MulAssign<f32> for Polynomial<'a> {
    fn mul_assign(&mut self, rhs: f32) {
        self.ops.iter_mut().for_each(|item| *item *= rhs);
    }
}
