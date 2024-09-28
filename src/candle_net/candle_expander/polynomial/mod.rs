use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt::Debug,
    hash::{BuildHasher, Hash},
    ops::{Mul, MulAssign},
};
mod indeterminant;
pub use indeterminant::*;
mod polycomponent;
pub use polycomponent::*;
mod variable;
use uuid::Uuid;
pub use variable::*;

#[derive(Debug, Clone, PartialEq)]
pub struct NewPolynomial<T> {
    ops: Vec<PolyComponent<T>>,
}

impl<T> Default for NewPolynomial<T> {
    fn default() -> Self {
        Self { ops: Vec::new() }
    }
}

impl<T> NewPolynomial<T> {
    pub fn parts(&self) -> &[PolyComponent<T>] {
        &self.ops
    }

    pub fn components(&self) -> &[PolyComponent<T>] {
        &self.ops
    }
    pub fn into_components(self) -> Vec<PolyComponent<T>> {
        self.ops
    }
}

impl<T: Debug + Hash + Eq> NewPolynomial<T> {
    pub fn map_operands<V: Debug + Clone, S: BuildHasher>(
        self,
        operands: &HashMap<T, V, S>,
    ) -> NewPolynomial<V> {
        NewPolynomial {
            ops: self
                .ops
                .into_iter()
                .map(|polyc| polyc.map_operands(operands))
                .collect(),
        }
    }
}

impl<T: Clone + PartialEq + PartialOrd + Ord + std::fmt::Debug> NewPolynomial<T> {
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    pub fn unit(var: T) -> Self {
        Self {
            ops: vec![PolyComponent::simple(1., var, 1)],
        }
    }

    pub fn add_indeterminate<I: Indeterminate<Variable = T>>(mut self, indeterminant: I) -> Self {
        todo!()
    }
    pub fn add_operation<I: Indeterminate<Variable = T>>(
        mut self,
        weight: f32,
        indeterminant: I,
        exponent: i32,
    ) -> Self {
        todo!()
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            ops: Vec::with_capacity(cap),
        }
    }
    pub fn with_operation(mut self, weight: f32, variable: T, exponent: i32) -> Self {
        self.handle_operation(weight, variable, exponent);
        self
    }

    pub fn with_polycomponent(mut self, component: PolyComponent<T>) -> Self {
        self.handle_polycomponent(component);
        self
    }

    pub fn handle_operation(&mut self, weight: f32, variable: T, exponent: i32) -> &mut Self {
        self.handle_polycomponent(PolyComponent::simple(weight, variable, exponent))
    }
    pub fn handle_polycomponent(&mut self, mut component: PolyComponent<T>) -> &mut Self {
        component.sort();
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

    pub fn sort_by_exponent(&mut self, order_on: T) {
        self.ops.iter_mut().for_each(|op| op.sort());

        self.ops.sort_by(|a, b| {
            let t_on_a = a.operands.iter().find(|op| op.var == order_on);
            let t_on_b = b.operands.iter().find(|op| op.var == order_on);

            match (t_on_a, t_on_b) {
                (Some(a), Some(b)) => a.exponent.cmp(&b.exponent),
                (Some(_), None) => Ordering::Greater,
                (None, Some(_)) => Ordering::Less,
                (None, None) => a
                    .weight()
                    .partial_cmp(&b.weight())
                    .unwrap_or(Ordering::Equal),
            }
        });
    }

    /// raises the whole polynomial to the power of -1.
    ///
    /// In turn, all of the exponents are multiplied by -1.
    pub fn invert(&mut self) {
        for component in self.ops.iter_mut() {
            for operand in component.operands.iter_mut() {
                operand.exponent *= -1;
            }
        }
    }

    /// FOIL
    fn mul_expand(self, other: &NewPolynomial<T>) -> NewPolynomial<T> {
        let mut result =
            NewPolynomial::with_capacity(self.components().len().max(other.components().len()) * 2); // a guesstimate

        for c1 in self.into_components() {
            for c2 in other.components() {
                let together = c1.clone() * c2.clone();
                result.handle_polycomponent(together);
            }
        }

        result
    }

    pub fn expand(&mut self, other: NewPolynomial<T>, weight: f32, exponent: i32) -> &mut Self {
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

impl<T> MulAssign<f32> for NewPolynomial<T> {
    fn mul_assign(&mut self, rhs: f32) {
        self.ops.iter_mut().for_each(|item| *item *= rhs);
    }
}
