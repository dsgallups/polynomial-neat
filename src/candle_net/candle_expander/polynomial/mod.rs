use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt::Debug,
    hash::{BuildHasher, Hash},
    ops::{Mul, MulAssign},
};
mod indeterminant;
use candle_core::Device;
pub use indeterminant::*;
mod polycomponent;
pub use polycomponent::*;
mod variable;
use uuid::Uuid;
pub use variable::*;

#[derive(Debug, Clone)]
pub struct Polynomial<'a, T> {
    device: &'a Device,
    ops: Vec<PolyComponent<T>>,
}

impl<'a, T> Polynomial<'a, T> {
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

impl<'a, T: Debug + Hash + Eq> Polynomial<'a, T> {
    pub fn map_operands<'b, V: Debug + Clone, S: BuildHasher>(
        self,
        operands: &'b HashMap<T, V, S>,
    ) -> Polynomial<V>
    where
        'a: 'b,
    {
        Polynomial {
            device: self.device,
            ops: self
                .ops
                .into_iter()
                .map(|polyc| polyc.map_operands(operands))
                .collect(),
        }
    }
}
impl<'dev, T> Polynomial<'dev, T> {
    pub fn new(device: &'dev Device) -> Self {
        Self {
            device,
            ops: Vec::new(),
        }
    }
    pub fn add_operation<'other, I: Indeterminate<'other, Variable = T>>(
        mut self,
        weight: f32,
        indeterminant: I,
        exponent: i32,
    ) -> Self {
        todo!()
    }

    pub fn with_device<'new>(self, device: &'new Device) -> Polynomial<'new, T> {
        Polynomial {
            device,
            ops: self.ops,
        }
    }

    pub fn add_indeterminate<'other, I: Indeterminate<'other, Variable = T>>(
        mut self,
        indeterminant: I,
    ) -> Self {
        todo!()
    }

    pub fn from_polycomponent(device: &'dev Device, component: PolyComponent<T>) -> Self {
        Self {
            device,
            ops: vec![component],
        }
    }
    pub fn unit(device: &'dev Device, var: T) -> Self {
        Self {
            device,
            ops: vec![PolyComponent::simple(1., var, 1)],
        }
    }
}

impl<'a, T: Clone + PartialEq + PartialOrd + Ord + std::fmt::Debug> Polynomial<'a, T> {
    pub fn with_capacity(device: &'a Device, cap: usize) -> Self {
        Self {
            device,
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
    fn mul_expand<'b>(self, other: &'b Polynomial<'b, T>) -> Polynomial<'a, T> {
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

impl<'a, T> MulAssign<f32> for Polynomial<'a, T> {
    fn mul_assign(&mut self, rhs: f32) {
        self.ops.iter_mut().for_each(|item| *item *= rhs);
    }
}
