use super::{PolyComponent, Polynomial};
use pretty_assertions::{assert_eq, assert_ne};
#[test]
pub fn add_simple_exponent() {
    let mut expander = Polynomial::default();

    expander.handle_operation(1., 1);
    expander.handle_operation(1., 1);
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
        .handle_operation(1., 1)
        .handle_operation(1., 2)
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
        .with_operation(1., 2)
        .with_operation(1., 1);

    //(f(x))^1 + 4x
    let mut flattened = Polynomial::default().with_operation(4., 1);

    flattened.expand(&binomial, 1., 1).sort_by_exponent();

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

#[test]
pub fn exponentiate_simple_monomial() {
    //x^2
    let monome = Polynomial::default().with_operation(1., 2);

    //(f(x))^3
    let mut flattened = Polynomial::default();
    flattened.expand(&monome, 1., 3).sort_by_exponent();

    let components = flattened.components();

    assert!(components.len() == 1);

    assert_eq!(
        components[0],
        PolyComponent {
            exponent: 6,
            weight: 1.
        }
    );
}

#[test]
pub fn exponentiate_weighted_monomial() {
    //3x^2
    let monome = Polynomial::default().with_operation(3., 2);

    //(f(x))^3
    let mut flattened = Polynomial::default();
    flattened.expand(&monome, 1., 3).sort_by_exponent();

    let components = flattened.components();

    assert!(components.len() == 1);

    assert_eq!(
        components[0],
        PolyComponent {
            exponent: 6,
            weight: 27.
        }
    );
}

#[test]
pub fn weight_and_exponentiate_weighted_monomial() {
    //6x^2
    let monome = Polynomial::default().with_operation(6., 2);

    //3(f(x))^3
    let mut flattened = Polynomial::default();
    flattened.expand(&monome, 3., 3).sort_by_exponent();

    let components = flattened.components();

    assert!(components.len() == 1);

    assert_eq!(
        components[0],
        PolyComponent {
            exponent: 6,
            weight: 648.
        }
    );
}

#[test]
pub fn exponentiate_simple_binomial() {
    //x^2 + x
    let binomial = Polynomial::default()
        .with_operation(1., 2)
        .with_operation(1., 1);

    //(f(x))^2
    let mut flattened = Polynomial::default();
    flattened.expand(&binomial, 1., 2).sort_by_exponent();

    let components = flattened.components();

    assert!(components.len() == 3);

    assert_eq!(
        components[0],
        PolyComponent {
            exponent: 2,
            weight: 1.
        }
    );
    assert_eq!(
        components[1],
        PolyComponent {
            exponent: 3,
            weight: 2.
        }
    );
    assert_eq!(
        components[2],
        PolyComponent {
            exponent: 4,
            weight: 1.
        }
    );
}

#[test]
pub fn exponentiate_complex_polynomial() {
    //x^2 + 4x + 3
    let binomial = Polynomial::default()
        .with_operation(1., 2)
        .with_operation(4., 1)
        .with_operation(3., 0);

    //(f(x))^3
    let mut flattened = Polynomial::default();
    flattened.expand(&binomial, 1., 3).sort_by_exponent();

    let components = flattened.components();

    assert!(components.len() == 7);

    assert_eq!(
        components[0],
        PolyComponent {
            exponent: 0,
            weight: 27.
        }
    );
    assert_eq!(
        components[1],
        PolyComponent {
            exponent: 1,
            weight: 108.
        }
    );
    assert_eq!(
        components[2],
        PolyComponent {
            exponent: 2,
            weight: 171.
        }
    );
    assert_eq!(
        components[3],
        PolyComponent {
            exponent: 3,
            weight: 136.
        }
    );
    assert_eq!(
        components[4],
        PolyComponent {
            exponent: 4,
            weight: 57.
        }
    );
    assert_eq!(
        components[5],
        PolyComponent {
            exponent: 5,
            weight: 12.
        }
    );

    assert_eq!(
        components[6],
        PolyComponent {
            exponent: 6,
            weight: 1.
        }
    );
}
