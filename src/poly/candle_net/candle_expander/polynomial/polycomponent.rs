use std::{
    collections::{BTreeMap, HashMap},
    fmt::Debug,
    hash::{BuildHasher, Hash},
    ops::{Mul, MulAssign},
};

use fnv::FnvHashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct PolyComponent {
    weight: f32,
    pub(crate) operands: BTreeMap<usize, i32>,
}

impl Default for PolyComponent {
    fn default() -> Self {
        Self {
            weight: 0.,
            operands: BTreeMap::new(),
        }
    }
}

impl PolyComponent {
    pub fn new() -> Self {
        Self {
            weight: 0.,
            operands: BTreeMap::new(),
        }
    }
    pub fn simple(weight: f32, var: usize, exponent: i32) -> Self {
        let mut btree = BTreeMap::new();
        if exponent == 0 {
            return Self {
                weight,
                operands: btree,
            };
        }

        btree.insert(var, exponent);

        Self {
            weight,
            operands: btree,
        }
    }
    pub fn weight(&self) -> f32 {
        self.weight
    }

    pub fn adjust_weight(&mut self, amt: f32) -> &mut Self {
        self.weight += amt;
        self
    }

    pub fn operands(&self) -> &BTreeMap<usize, i32> {
        &self.operands
    }
    pub fn powi(mut self, exponent: i32) -> Self {
        self.weight = self.weight.powi(exponent);
        self.operands.iter_mut().for_each(|(i, exp)| {
            *exp *= exponent;
        });

        self
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Adds the operand to the component. Simplifies if the operand already exists and sorts.
    pub fn with_operand(mut self, variable: usize, exponent: i32) -> Self {
        if exponent == 0 {
            return self;
        }

        let mut remove = false;
        self.operands
            .entry(variable)
            .and_modify(|exp| {
                *exp += exponent;
                if *exp == 0 {
                    remove = true
                }
            })
            .or_insert(exponent);
        if remove {
            self.operands.remove(&variable);
        }

        self
    }

    pub fn base(weight: f32) -> Self {
        Self {
            weight,
            operands: BTreeMap::new(),
        }
    }

    /// Note: does not simplify duplicates. use `with_operand` for this behavior.
    pub fn from_raw_parts(weight: f32, operands: BTreeMap<usize, i32>) -> Self {
        Self { weight, operands }
    }
}

// should work the same way as 4x^0 is handled.
// this is just efficient.
impl MulAssign<f32> for PolyComponent {
    fn mul_assign(&mut self, rhs: f32) {
        self.weight *= rhs;
    }
}
impl MulAssign for PolyComponent {
    fn mul_assign(&mut self, rhs: Self) {
        self.weight *= rhs.weight;
        for (rhs_var, rhs_exp) in rhs.operands {
            let remove = match self.operands.get_mut(&rhs_var) {
                Some(exp) => {
                    *exp += rhs_exp;
                    *exp == 0
                }
                None => {
                    self.operands.insert(rhs_var, rhs_exp);
                    false
                }
            };
            if remove {
                self.operands.remove(&rhs_var);
            }
        }
    }
}

impl Mul for PolyComponent {
    type Output = PolyComponent;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut new_ops = self.operands;
        for (rhs_var, rhs_exp) in rhs.operands {
            let remove = match new_ops.get_mut(&rhs_var) {
                Some(exp) => {
                    *exp += rhs_exp;
                    *exp == 0
                }
                None => {
                    new_ops.insert(rhs_var, rhs_exp);
                    false
                }
            };
            if remove {
                new_ops.remove(&rhs_var);
            }
        }
        PolyComponent {
            operands: new_ops,
            weight: self.weight * rhs.weight,
        }
    }
}
