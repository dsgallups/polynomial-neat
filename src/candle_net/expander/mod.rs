use std::{
    cmp::Ordering,
    ops::{Mul, MulAssign},
};

use uuid::Uuid;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Variable<T> {
    var: T,
    exponent: i32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolyComponent<T> {
    weight: f32,
    operands: Vec<Variable<T>>,
}

impl<T> PolyComponent<T> {
    pub fn new(weight: f32, var: T, exponent: i32) -> Self {
        if exponent == 0 {
            return Self {
                weight,
                operands: Vec::new(),
            };
        }

        Self {
            weight,
            operands: vec![Variable { var, exponent }],
        }
    }
}

// should work the same way as 4x^0 is handled.
// this is just efficient.
impl<T> MulAssign<f32> for PolyComponent<T> {
    fn mul_assign(&mut self, rhs: f32) {
        self.weight *= rhs;
    }
}
impl<T: PartialEq> MulAssign for PolyComponent<T> {
    fn mul_assign(&mut self, rhs: Self) {
        self.weight *= rhs.weight;
        for operand in rhs.operands {
            match self.operands.iter_mut().find(|op| op.var == operand.var) {
                Some(op) => {
                    op.exponent += operand.exponent;
                }
                None => self.operands.push(operand),
            }
        }
    }
}

impl<T: PartialEq> Mul for PolyComponent<T> {
    type Output = PolyComponent<T>;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut new_ops = self.operands;
        for operand in rhs.operands {
            match new_ops.iter_mut().find(|op| op.var == operand.var) {
                Some(op) => {
                    op.exponent += operand.exponent;
                }
                None => new_ops.push(operand),
            }
        }
        PolyComponent {
            operands: new_ops,
            weight: self.weight * rhs.weight,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Polynomial<T> {
    ops: Vec<PolyComponent<T>>,
}

impl<T: Clone + PartialEq + PartialOrd + std::fmt::Debug> Polynomial<T> {
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    pub fn unit(var: T) -> Self {
        Self {
            ops: vec![PolyComponent::new(1., var, 1)],
        }
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
    pub fn handle_operation(&mut self, weight: f32, variable: T, exponent: i32) -> &mut Self {
        self.handle_polycomponent(PolyComponent::new(weight, variable, exponent))
    }
    pub fn handle_polycomponent(&mut self, component: PolyComponent<T>) -> &mut Self {
        match self
            .ops
            .iter_mut()
            .find(|op| op.operands == component.operands)
        {
            Some(op) => {
                op.weight += component.weight;
            }
            None => self.ops.push(component),
        }
        self
    }

    pub fn sort_by_exponent(&mut self, order_on: T) {
        self.ops.sort_by(|a, b| {
            let t_on_a = a.operands.iter().find(|op| op.var == order_on);
            let t_on_b = b.operands.iter().find(|op| op.var == order_on);

            match (t_on_a, t_on_b) {
                (Some(a), Some(b)) => a.exponent.cmp(&b.exponent),
                (Some(_), None) => Ordering::Greater,
                (None, Some(_)) => Ordering::Less,
                (None, None) => a.weight.partial_cmp(&b.weight).unwrap_or(Ordering::Equal),
            }
        });
    }

    pub fn components(&self) -> &[PolyComponent<T>] {
        &self.ops
    }
    pub fn into_components(self) -> Vec<PolyComponent<T>> {
        self.ops
    }

    fn mul_expand(self, other: &Polynomial<T>) -> Polynomial<T> {
        let mut result =
            Polynomial::with_capacity(self.components().len().max(other.components().len()) * 2); // a guesstimate

        for c1 in self.into_components() {
            for c2 in other.components() {
                let together = c1.clone() * c2.clone();
                result.handle_polycomponent(together);
            }
        }

        result
    }

    pub fn expand(&mut self, other: Polynomial<T>, weight: f32, exponent: i32) -> &mut Self {
        // important to clone here since mutating other will multiply the exponents.
        let mut running = other.clone();
        if exponent > 1 {
            for _ in 1..exponent {
                running = running.mul_expand(&other);
            }
        }

        running *= weight;

        for component in running.into_components() {
            self.handle_polycomponent(component);
        }

        self
    }
}

impl<T> MulAssign<f32> for Polynomial<T> {
    fn mul_assign(&mut self, rhs: f32) {
        self.ops.iter_mut().for_each(|item| *item *= rhs);
    }
}
