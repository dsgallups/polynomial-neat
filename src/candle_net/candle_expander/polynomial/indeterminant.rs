use candle_core::Device;

use super::{PolyComponent, Polynomial};

/*
    3 * (X) ^ 2 -> 3x^2 = Variable -> Polynomial
    3 * (3X^2) -> 9x^2 = Polycomponent -> Polynomial

*/
pub trait Indeterminate<'dev> {
    type Variable;
    // applies a weight and exponent to self
    fn apply_operation(
        self,
        device: &'dev Device,
        weight: f32,
        exponent: i32,
    ) -> Polynomial<Self::Variable>;
    fn identity(self, device: &'dev Device) -> Polynomial<'dev, Self::Variable>;
}

impl<'dev, 'dev2, T> Indeterminate<'dev> for PolyComponent<T>
where
    T: Indeterminate<'dev2>,
{
    type Variable = T;
    fn apply_operation(
        self,
        device: &'dev Device,
        weight: f32,
        exponent: i32,
    ) -> Polynomial<'dev, Self::Variable> {
        let mut exponentiated = self.powi(exponent);
        exponentiated *= weight;

        Polynomial::from_polycomponent(device, exponentiated)
    }
    fn identity(self, device: &'dev Device) -> Polynomial<'dev, Self::Variable> {
        Polynomial::from_polycomponent(device, self)
    }
}

impl<'dev, 'dev2, 'dev3, T> Indeterminate<'dev> for Polynomial<'dev3, T>
where
    T: Indeterminate<'dev2>,
{
    type Variable = T;
    fn apply_operation(
        self,
        device: &'dev Device,
        weight: f32,
        exponent: i32,
    ) -> Polynomial<'dev, Self::Variable> {
        todo!();
    }
    fn identity(self, device: &'dev Device) -> Polynomial<'dev, Self::Variable> {
        self.with_device(device)
    }
}
