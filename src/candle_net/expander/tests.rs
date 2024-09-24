use super::{PolyComponent, Polynomial};
use pretty_assertions::{assert_eq, assert_ne};
#[test]
pub fn add_simple_exponent() {
    let mut expander = Polynomial::default();

    expander.handle_operation(1, 1.);
    expander.handle_operation(1, 1.);
    let components = expander.components();

    assert!(components.len() == 1);
    assert_eq!(
        components[0],
        PolyComponent {
            exponent: 1,
            weight: 2.
        }
    )
}

#[test]
pub fn add_separate_exponents() {
    let mut expander = Polynomial::default();
    expander
        .handle_operation(1, 1.)
        .handle_operation(2, 1.)
        .sort_by_exponent();

    let components = expander.components();

    assert!(components.len() == 2);
    assert_eq!(
        components[0],
        PolyComponent {
            exponent: 1,
            weight: 1.
        }
    );
    assert_eq!(
        components[0],
        PolyComponent {
            exponent: 2,
            weight: 1.
        }
    )
}

#[test]
pub fn add_binomial() {
    //x^2 + x
    let binomial = Polynomial::default()
        .with_operation(2, 1.)
        .with_operation(1, 1.);

    //f(x)^2 + 4x

    let flattened = Polynomial::default()
        .with_operation(1, 1.)
        .expand(&binomial, 2, 1.);
}
