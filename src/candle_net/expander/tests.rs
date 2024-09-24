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

    println!("components: {:?}", components);

    assert_eq!(
        components[0],
        PolyComponent {
            exponent: 1,
            weight: 1.
        }
    );
    assert_eq!(
        components[1],
        PolyComponent {
            exponent: 2,
            weight: 1.
        }
    )
}

#[test]
pub fn add_simple_binomial() {
    //x^2 + x
    let binomial = Polynomial::default()
        .with_operation(2, 1.)
        .with_operation(1, 1.);

    //(f(x))^1 + 4x
    let mut flattened = Polynomial::default().with_operation(1, 4.);
    flattened.expand(&binomial, 1, 1.).sort_by_exponent();

    // should be x^2 + 5x
    let components = flattened.components();

    assert!(components.len() == 2);

    assert_eq!(
        components[0],
        PolyComponent {
            exponent: 1,
            weight: 5.
        }
    );

    assert_eq!(
        components[1],
        PolyComponent {
            exponent: 2,
            weight: 1.
        }
    )
}
