use std::ops::{Mul, MulAssign};

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolyComponent {
    weight: f32,
    exponent: i32,
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
        self.exponent += rhs.exponent;
        self.weight *= rhs.weight;
    }
}

impl Mul for PolyComponent {
    type Output = PolyComponent;
    fn mul(self, rhs: Self) -> Self::Output {
        PolyComponent {
            exponent: self.exponent + rhs.exponent,
            weight: self.weight * rhs.weight,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Polynomial {
    ops: Vec<PolyComponent>,
}

impl Polynomial {
    pub fn with_operation(mut self, weight: f32, exponent: i32) -> Self {
        self.handle_operation(weight, exponent);
        self
    }
    pub fn handle_operation(&mut self, weight: f32, exponent: i32) -> &mut Self {
        self.handle_polycomponent(PolyComponent { exponent, weight })
    }
    pub fn handle_polycomponent(&mut self, component: PolyComponent) -> &mut Self {
        match self
            .ops
            .iter_mut()
            .find(|op| op.exponent == component.exponent)
        {
            Some(op) => {
                op.weight += component.weight;
            }
            None => self.ops.push(component),
        }
        self
    }

    pub fn sort_by_exponent(&mut self) {
        self.ops.sort_by(|a, b| a.exponent.cmp(&b.exponent));
    }

    pub fn components(&self) -> &[PolyComponent] {
        &self.ops
    }
    pub fn into_components(self) -> Vec<PolyComponent> {
        self.ops
    }

    pub fn mul_expand(&self, other: &Polynomial) -> Polynomial {
        let mut result = Polynomial::default();

        for c1 in self.components() {
            for c2 in other.components() {
                result.handle_polycomponent(*c1 * *c2);
            }
        }

        result
    }

    pub fn expand(&mut self, other: &Polynomial, weight: f32, exponent: i32) -> &mut Self {
        let mut running_poly = other.clone();

        if exponent > 1 {
            for _ in 1..exponent {
                running_poly = running_poly.mul_expand(other);
            }
        }

        running_poly *= weight;

        for component in running_poly.into_components() {
            self.handle_polycomponent(component);
        }

        self
    }
}

impl MulAssign<f32> for Polynomial {
    fn mul_assign(&mut self, rhs: f32) {
        self.ops.iter_mut().for_each(|item| *item *= rhs);
    }
}
