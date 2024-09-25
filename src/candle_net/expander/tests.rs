use crate::candle_net::expander::Variable;

use super::{PolyComponent, Polynomial};
use pretty_assertions::{assert_eq, assert_ne};

#[derive(Clone, Copy, PartialOrd, Ord, Debug, PartialEq, Default, Eq)]
struct X;

#[test]
pub fn add_simple_exponent() {
    let mut expander = Polynomial::default();

    expander.handle_operation(1., X, 1);
    expander.handle_operation(1., X, 1);
    let components = expander.components();

    assert!(components.len() == 1);
    assert_eq!(components[0], PolyComponent::new(2., X, 1))
}

#[test]
pub fn add_separate_exponents() {
    let mut expander = Polynomial::default();
    expander
        .handle_operation(1., X, 1)
        .handle_operation(1., X, 2)
        .sort_by_exponent(X);

    let components = expander.components();

    assert_eq!(components[0], PolyComponent::new(1., X, 1));
    assert_eq!(components[1], PolyComponent::new(1., X, 2),)
}

#[test]
pub fn add_simple_binomial() {
    //x^2 + x
    let binomial = Polynomial::default()
        .with_operation(1., X, 2)
        .with_operation(1., X, 1);

    //(f(x))^1 + 4x
    let mut flattened = Polynomial::default().with_operation(4., X, 1);

    flattened.expand(binomial, 1., 1).sort_by_exponent(X);

    // should be x^2 + 5x
    let components = flattened.components();

    assert!(components.len() == 2);

    assert_eq!(components[0], PolyComponent::new(5., X, 1));

    assert_eq!(components[1], PolyComponent::new(1., X, 2),)
}

#[test]
pub fn exponentiate_simple_monomial() {
    //x^2
    let monome = Polynomial::default().with_operation(1., X, 2);

    //(f(x))^3
    let mut flattened = Polynomial::default();
    flattened.expand(monome, 1., 3).sort_by_exponent(X);

    let components = flattened.components();

    assert!(components.len() == 1);

    assert_eq!(components[0], PolyComponent::new(1., X, 6));
}

#[test]
pub fn exponentiate_weighted_monomial() {
    //3x^2
    let monome = Polynomial::default().with_operation(3., X, 2);

    //(f(x))^3
    let mut flattened = Polynomial::default();
    flattened.expand(monome, 1., 3).sort_by_exponent(X);

    let components = flattened.components();

    assert!(components.len() == 1);

    assert_eq!(components[0], PolyComponent::new(27., X, 6),);
}

#[test]
pub fn weight_and_exponentiate_weighted_monomial() {
    //6x^2
    let monome = Polynomial::default().with_operation(6., X, 2);

    //3(f(x))^3
    let mut flattened = Polynomial::default();
    flattened.expand(monome, 3., 3).sort_by_exponent(X);

    let components = flattened.components();

    assert!(components.len() == 1);

    assert_eq!(components[0], PolyComponent::new(648., X, 6));
}

#[test]
pub fn exponentiate_simple_binomial() {
    //x^2 + x
    let binomial = Polynomial::default()
        .with_operation(1., X, 2)
        .with_operation(1., X, 1);

    //(f(x))^2
    let mut flattened = Polynomial::default();
    flattened.expand(binomial, 1., 2).sort_by_exponent(X);

    let components = flattened.components();

    assert!(components.len() == 3);

    assert_eq!(components[0], PolyComponent::new(1., X, 2),);
    assert_eq!(components[1], PolyComponent::new(2., X, 3),);
    assert_eq!(components[2], PolyComponent::new(1., X, 4),);
}

#[test]
pub fn exponentiate_complex_polynomial() {
    //x^2 + 4x + 3
    let binomial = Polynomial::default()
        .with_operation(1., X, 2)
        .with_operation(4., X, 1)
        .with_operation(3., X, 0);

    //(f(x))^3
    let mut flattened = Polynomial::default();
    flattened.expand(binomial, 1., 3).sort_by_exponent(X);

    let components = flattened.components();

    assert!(components.len() == 7);

    assert_eq!(components[0], PolyComponent::new(27., X, 0),);
    assert_eq!(components[1], PolyComponent::new(108., X, 1),);
    assert_eq!(components[2], PolyComponent::new(171., X, 2),);
    assert_eq!(components[3], PolyComponent::new(136., X, 3));
    assert_eq!(components[4], PolyComponent::new(57., X, 4),);
    assert_eq!(components[5], PolyComponent::new(12., X, 5),);
    assert_eq!(components[6], PolyComponent::new(1., X, 6),);
}

#[derive(Clone, Copy, PartialOrd, Ord, Debug, PartialEq, Eq)]
enum V {
    X,
    Y,
}

#[test]
pub fn add_simple_exponent_v() {
    let mut expander = Polynomial::default();

    expander.handle_operation(1., V::X, 1);
    expander.handle_operation(1., V::Y, 1);
    let components = expander.components();

    assert!(components.len() == 2);
    assert_eq!(components[0], PolyComponent::new(1., V::X, 1));
    assert_eq!(components[1], PolyComponent::new(1., V::Y, 1));
}

#[test]
pub fn multiply_multi_component() {
    let monome = Polynomial::default().with_polycomponent(PolyComponent::new_complex(
        2.,
        vec![Variable::new(V::X, 1), Variable::new(V::Y, 1)],
    ));

    let mut simple_outer = Polynomial::default();
    simple_outer.expand(monome, 3., 2);

    let components = simple_outer.components();

    assert!(components.len() == 1);
    assert_eq!(
        components[0],
        PolyComponent::new_complex(12., vec![Variable::new(V::X, 2), Variable::new(V::Y, 2)])
    );
}

/* Negative exponents */

#[test]
pub fn neg_exponentiate_monomial() {
    //x^2
    let monome = Polynomial::default().with_operation(1., X, 2);

    //(f(x))^-3
    let mut flattened = Polynomial::default();
    flattened.expand(monome, 1., -3).sort_by_exponent(X);

    let components = flattened.components();

    assert!(components.len() == 1);

    assert_eq!(components[0], PolyComponent::new(1., X, -6));
}

#[test]
pub fn neg_exponentiate_binomial() {
    //x^2 + x
    let binomial = Polynomial::default()
        .with_operation(1., X, 2)
        .with_operation(1., X, 1);

    //2 * (f(x))^-2
    let mut flattened = Polynomial::default();
    flattened.expand(binomial, 2., -2).sort_by_exponent(X);

    let components = flattened.components();

    assert!(components.len() == 3);

    assert_eq!(components[0], PolyComponent::new(2., X, -4));
    assert_eq!(components[1], PolyComponent::new(4., X, -3));
    assert_eq!(components[2], PolyComponent::new(2., X, -2));
}
