use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt::{self, Debug},
    hash::{BuildHasher, Hash},
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

impl<T: fmt::Display> fmt::Display for Variable<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.exponent == 1 {
            write!(f, "[{}]", self.var)
        } else {
            write!(f, "[{}]^{}", self.var, self.exponent)
        }
    }
}

impl<T> Variable<T> {
    pub fn new(var: T, exponent: i32) -> Self {
        Self { var, exponent }
    }

    pub fn exponent(&self) -> i32 {
        self.exponent
    }
    pub fn var(&self) -> &T {
        &self.var
    }
}

impl<T: Debug + Hash + Eq> Variable<T> {
    pub fn map_operands<V: Clone + Debug, S: BuildHasher>(
        self,
        operands: &HashMap<T, V, S>,
    ) -> Variable<V> {
        let Some(new_var) = operands.get(&self.var).cloned() else {
            panic!("couldn't find {:?}\noperands: {:#?}", self.var, operands);
        };

        Variable {
            var: new_var,
            exponent: self.exponent,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolyComponent<T> {
    weight: f32,
    pub(crate) operands: Vec<Variable<T>>,
}

impl<T: fmt::Display> fmt::Display for PolyComponent<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.weight != 1. {
            write!(f, "{}", self.weight)?;
        }

        if self.operands().len() == 1 {
            let singleton = self.operands().first().unwrap();
            write!(f, "{singleton}")?;
        } else {
            for operand in self.operands().iter() {
                write!(f, "({operand})")?;
            }
        }

        Ok(())
    }
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
    pub fn weight(&self) -> f32 {
        self.weight
    }

    pub fn operands(&self) -> &[Variable<T>] {
        &self.operands
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

impl<T: fmt::Display> fmt::Display for Polynomial<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, op) in self.ops.iter().enumerate() {
            write!(f, "{op}")?;
            if i != self.ops.len() - 1 {
                write!(f, " + ")?;
            }
        }
        Ok(())
    }
}

impl<T> Default for Polynomial<T> {
    fn default() -> Self {
        Self { ops: Vec::new() }
    }
}

impl<T> Polynomial<T> {
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

impl<T: Debug + Hash + Eq> Polynomial<T> {
    pub fn map_operands<V: Debug + Clone, S: BuildHasher>(
        self,
        operands: &HashMap<T, V, S>,
    ) -> Polynomial<V> {
        Polynomial {
            ops: self
                .ops
                .into_iter()
                .map(|polyc| polyc.map_operands(operands))
                .collect(),
        }
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
        self.ops.iter_mut().for_each(|op| op.sort());

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
