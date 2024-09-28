use std::{
    collections::HashMap,
    fmt::Debug,
    hash::{BuildHasher, Hash},
    ops::{Mul, MulAssign},
};

use super::Variable;

#[derive(Debug, Clone, PartialEq)]
pub struct PolyComponent<T> {
    weight: f32,
    pub(crate) operands: Vec<Variable<T>>,
}

impl<T> Default for PolyComponent<T> {
    fn default() -> Self {
        Self {
            weight: 0.,
            operands: Vec::new(),
        }
    }
}

impl<T> PolyComponent<T> {
    pub fn new() -> Self {
        Self {
            weight: 0.,
            operands: Vec::new(),
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
            operands: vec![Variable::new(var, exponent)],
        }
    }
    pub fn weight(&self) -> f32 {
        self.weight
    }

    pub fn adjust_weight(&mut self, amt: f32) -> &mut Self {
        self.weight += amt;
        self
    }

    pub fn operands(&self) -> &[Variable<T>] {
        &self.operands
    }
    pub fn powi(mut self, exponent: i32) -> Self {
        self.weight = self.weight.powi(exponent);
        self.operands.iter_mut().for_each(|var| {
            var.exponent *= exponent;
        });

        self
    }
}

impl<T: Debug + Hash + Eq> PolyComponent<T> {
    pub fn map_operands<V: Clone + Debug, S: BuildHasher>(
        self,
        operands: &HashMap<T, V, S>,
    ) -> PolyComponent<V> {
        PolyComponent {
            weight: self.weight,
            operands: self
                .operands
                .into_iter()
                .map(|var| var.map_operands(operands))
                .collect(),
        }
    }
}

impl<T: Ord> PolyComponent<T> {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            weight: 0.,
            operands: Vec::with_capacity(cap),
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
