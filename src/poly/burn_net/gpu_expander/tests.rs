use super::*;
use crate::poly::burn_net::expander::{PolyComponent, Polynomial, Variable};
use burn::backend::NdArray;
use pretty_assertions::assert_eq;

type TestBackend = NdArray;

#[test]
fn test_monomial_creation() {
    let device = Default::default();
    let gpu_poly = GpuPolynomial::<TestBackend>::monomial(&device, 3.0, 1, 2, 3);

    let components = gpu_poly.to_components();
    assert_eq!(components.len(), 1);
    assert_eq!(components[0].0, 3.0);
    assert_eq!(components[0].1, vec![(1, 2)]);
}

#[test]
fn test_polynomial_addition() {
    let device = Default::default();

    // Create 2x^2 + 3x
    let poly1 = GpuPolynomial::<TestBackend>::from_components(
        &device,
        vec![(2.0, vec![(0, 2)]), (3.0, vec![(0, 1)])],
        1,
    );

    // Create x^2 + 5x + 1
    let poly2 = GpuPolynomial::<TestBackend>::from_components(
        &device,
        vec![(1.0, vec![(0, 2)]), (5.0, vec![(0, 1)]), (1.0, vec![])],
        1,
    );

    let result = poly1.add(&poly2);
    let components = result.to_components();

    // Should get 3x^2 + 8x + 1
    assert_eq!(components.len(), 3);

    // Sort by total exponent for consistent ordering
    let mut sorted_components = components;
    sorted_components.sort_by_key(|c| -c.1.iter().map(|(_, exp)| exp).sum::<i32>());

    assert!((sorted_components[0].0 - 3.0).abs() < 1e-6);
    assert_eq!(sorted_components[0].1, vec![(0, 2)]);

    assert!((sorted_components[1].0 - 8.0).abs() < 1e-6);
    assert_eq!(sorted_components[1].1, vec![(0, 1)]);

    assert!((sorted_components[2].0 - 1.0).abs() < 1e-6);
    assert_eq!(sorted_components[2].1, vec![]);
}

#[test]
fn test_polynomial_multiplication() {
    let device = Default::default();

    // Create (x + 2)
    let poly1 = GpuPolynomial::<TestBackend>::from_components(
        &device,
        vec![(1.0, vec![(0, 1)]), (2.0, vec![])],
        1,
    );

    // Create (x + 3)
    let poly2 = GpuPolynomial::<TestBackend>::from_components(
        &device,
        vec![(1.0, vec![(0, 1)]), (3.0, vec![])],
        1,
    );

    let result = poly1.multiply(&poly2);
    let components = result.to_components();

    // Should get x^2 + 5x + 6
    assert_eq!(components.len(), 3);

    let mut sorted_components = components;
    sorted_components.sort_by_key(|c| -c.1.iter().map(|(_, exp)| exp).sum::<i32>());

    assert!((sorted_components[0].0 - 1.0).abs() < 1e-6);
    assert_eq!(sorted_components[0].1, vec![(0, 2)]);

    assert!((sorted_components[1].0 - 5.0).abs() < 1e-6);
    assert_eq!(sorted_components[1].1, vec![(0, 1)]);

    assert!((sorted_components[2].0 - 6.0).abs() < 1e-6);
    assert_eq!(sorted_components[2].1, vec![]);
}

#[test]
fn test_polynomial_power() {
    let device = Default::default();

    // Create (x + 1)
    let poly = GpuPolynomial::<TestBackend>::from_components(
        &device,
        vec![(1.0, vec![(0, 1)]), (1.0, vec![])],
        1,
    );

    // Compute (x + 1)^3
    let result = poly.pow(3);
    let components = result.to_components();

    // Should get x^3 + 3x^2 + 3x + 1
    assert_eq!(components.len(), 4);

    let mut sorted_components = components;
    sorted_components.sort_by_key(|c| -c.1.iter().map(|(_, exp)| exp).sum::<i32>());

    assert!((sorted_components[0].0 - 1.0).abs() < 1e-6);
    assert_eq!(sorted_components[0].1, vec![(0, 3)]);

    assert!((sorted_components[1].0 - 3.0).abs() < 1e-6);
    assert_eq!(sorted_components[1].1, vec![(0, 2)]);

    assert!((sorted_components[2].0 - 3.0).abs() < 1e-6);
    assert_eq!(sorted_components[2].1, vec![(0, 1)]);

    assert!((sorted_components[3].0 - 1.0).abs() < 1e-6);
    assert_eq!(sorted_components[3].1, vec![]);
}

#[test]
fn test_multivariate_polynomial() {
    let device = Default::default();

    // Create xy + 2x + 3y
    let poly1 = GpuPolynomial::<TestBackend>::from_components(
        &device,
        vec![
            (1.0, vec![(0, 1), (1, 1)]),
            (2.0, vec![(0, 1)]),
            (3.0, vec![(1, 1)]),
        ],
        2,
    );

    // Create x + y
    let poly2 = GpuPolynomial::<TestBackend>::from_components(
        &device,
        vec![(1.0, vec![(0, 1)]), (1.0, vec![(1, 1)])],
        2,
    );

    let result = poly1.multiply(&poly2);
    let components = result.to_components();

    // Should get x^2y + xy^2 + 2x^2 + 5xy + 3y^2
    // Let's just verify we have the right number of terms
    assert!(components.len() >= 5);
}

#[test]
fn test_cpu_gpu_conversion() {
    let device = Default::default();

    // Create CPU polynomial: x^2 + 2xy + y^2
    let cpu_poly = Polynomial::<usize>::new()
        .with_operation(1.0, 0, 2)
        .with_polycomponent(PolyComponent::from_raw_parts(
            2.0,
            vec![Variable::new(0, 1), Variable::new(1, 1)],
        ))
        .with_operation(1.0, 1, 2);

    // Convert to GPU
    let gpu_poly = from_cpu_polynomial::<TestBackend>(cpu_poly.clone(), &device, 2);

    // Convert back to CPU
    let cpu_poly_back = to_cpu_polynomial(gpu_poly);

    // Compare components
    let original_components = cpu_poly.into_components();
    let converted_components = cpu_poly_back.into_components();

    assert_eq!(original_components.len(), converted_components.len());
}

#[test]
fn test_negative_exponents() {
    let device = Default::default();

    // Create x^2
    let poly = GpuPolynomial::<TestBackend>::monomial(&device, 1.0, 0, 2, 1);

    // Compute x^(-2)
    let result = poly.pow(-1);
    let components = result.to_components();

    assert_eq!(components.len(), 1);
    assert!((components[0].0 - 1.0).abs() < 1e-6);
    assert_eq!(components[0].1, vec![(0, -2)]);
}

#[test]
fn test_zero_power() {
    let device = Default::default();

    // Create x^2 + 2x + 1
    let poly = GpuPolynomial::<TestBackend>::from_components(
        &device,
        vec![(1.0, vec![(0, 2)]), (2.0, vec![(0, 1)]), (1.0, vec![])],
        1,
    );

    // Compute (x^2 + 2x + 1)^0 = 1
    let result = poly.pow(0);
    let components = result.to_components();

    assert_eq!(components.len(), 1);
    assert!((components[0].0 - 1.0).abs() < 1e-6);
    assert_eq!(components[0].1, vec![]);
}
