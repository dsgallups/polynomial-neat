use candle_core::Device;

use super::{PolyComponent, Polynomial};

/*
    3 * (X) ^ 2 -> 3x^2 = Variable -> Polynomial
    3 * (3X^2) -> 9x^2 = Polycomponent -> Polynomial

*/
pub trait Indeterminate<'dev> {
    /// applies a weight and exponent to self
    /// The returned value should be SIMPLIFIED.
    fn apply_operation(self, device: &'dev Device, weight: f32, exponent: i32) -> Polynomial<'dev>;
    fn identity(self, device: &'dev Device) -> Polynomial<'dev>;
}

impl<'dev> Indeterminate<'dev> for PolyComponent {
    fn apply_operation(self, device: &'dev Device, weight: f32, exponent: i32) -> Polynomial<'dev> {
        let mut exponentiated = self.powi(exponent);
        exponentiated *= weight;

        Polynomial::from_polycomponent(device, exponentiated)
    }
    fn identity(self, device: &'dev Device) -> Polynomial<'dev> {
        Polynomial::from_polycomponent(device, self)
    }
}

impl<'dev, 'dev2> Indeterminate<'dev> for Polynomial<'dev2> {
    fn apply_operation(self, device: &'dev Device, weight: f32, exponent: i32) -> Polynomial<'dev> {
        let all_variables = self.variadics();
        let expanded = self
            .into_components()
            .into_iter()
            .map(|component| {
                let coef = component.weight();
                let variable_exponents = vec![0.; all_variables.len()];
                let mut variable_exponents = Vec::with_capacity(all_variables.len());
                for (op_var, op_exp) in component.operands() {
                    let variable_position = all_variables
                        .iter()
                        .position(|var_ag| op_var == var_ag)
                        .unwrap();
                    variable_exponents[variable_position] = *op_exp
                }

                (coef, variable_exponents)
            })
            .collect::<Vec<_>>();

        //for

        todo!();
    }
    fn identity(self, device: &'dev Device) -> Polynomial<'dev> {
        self.with_device(device)
    }
}
