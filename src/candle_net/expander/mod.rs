use std::{
    cmp::Ordering,
    ops::{Mul, MulAssign},
};

use uuid::Uuid;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
pub struct Variable<T> {
    var: T,
    exponent: i32,
}

impl<T> Variable<T> {
    pub fn new(var: T, exponent: i32) -> Self {
        Self { var, exponent }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolyComponent<T> {
    weight: f32,
    operands: Vec<Variable<T>>,
}

impl<T> Default for PolyComponent<T> {
    fn default() -> Self {
        Self {
            weight: 0.,
            operands: Vec::new(),
        }
    }
}

impl<T: Ord> PolyComponent<T> {
    pub fn new() -> Self {
        Self {
            weight: 0.,
            operands: Vec::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            weight: 0.,
            operands: Vec::with_capacity(cap),
        }
    }

    pub fn simple(weight: f32, var: T, exponent: i32) -> Self {
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

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Adds the operand to the component. Simplifies if the operand already exists and sorts.
    pub fn with_operand(mut self, var: T, exponent: i32) -> Self {
        if exponent == 0 {
            return self;
        }

        match self.operands.iter_mut().find(|op| op.var == var) {
            Some(op) => {
                op.exponent += exponent;
            }
            None => {
                self.operands.push(Variable { var, exponent });
                self.operands.sort();
                return self;
            }
        }

        self.operands.retain(|op| op.exponent != 0);
        self
    }

    pub fn base(weight: f32) -> Self {
        Self {
            weight,
            operands: Vec::new(),
        }
    }

    /// Note: does not simplify duplicates. use `with_operand` for this behavior.
    pub fn from_raw_parts(weight: f32, mut operands: Vec<Variable<T>>) -> Self {
        operands.sort();

        Self { weight, operands }
    }

    pub fn sort(&mut self) {
        self.operands.sort();
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

#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial<T> {
    ops: Vec<PolyComponent<T>>,
}

impl<T> Default for Polynomial<T> {
    fn default() -> Self {
        Self { ops: Vec::new() }
    }
}

impl<T: Clone + PartialEq + PartialOrd + Ord + std::fmt::Debug> Polynomial<T> {
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    pub fn unit(var: T) -> Self {
        Self {
            ops: vec![PolyComponent::simple(1., var, 1)],
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

impl<T> MulAssign<f32> for Polynomial<T> {
    fn mul_assign(&mut self, rhs: f32) {
        self.ops.iter_mut().for_each(|item| *item *= rhs);
    }
}
