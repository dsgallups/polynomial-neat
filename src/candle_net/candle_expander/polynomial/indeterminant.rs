use super::NewPolynomial;

/*
    3 * (X) ^ 2 -> 3x^2 = Variable -> Polynomial
    3 * (3X^2) -> 9x^2 = Polycomponent -> Polynomial

*/
pub trait Indeterminate {
    type Variable;
    // applies a weight and exponent to self
    fn apply_operation(self, weight: f32, exponent: i32) -> NewPolynomial<Self::Variable>;
    fn identity(self) -> NewPolynomial<Self::Variable>;
}
