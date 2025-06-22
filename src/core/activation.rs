//! Activation function components for polynomial neural networks.
//!
//! This module provides the building blocks for polynomial activation functions used in
//! the evolved neural networks. The polynomial activation function for each neuron is
//! computed as:
//!
//! ```text
//! output = Σ(weight_i * input_i^exponent_i) + bias
//! ```
//!
//! Where:
//! - `weight_i` is the connection weight from input i
//! - `input_i` is the value from input neuron i
//! - `exponent_i` is the evolved exponent for that input
//! - `bias` is the neuron's bias term
//!
//! # Components
//!
//! - [`Bias`]: Represents the bias term added to each neuron's output
//! - [`Exponent`]: Represents the exponent applied to each input value
//!
//! # Example
//!
//! ```
//! use polynomial_neat::core::activation::{Bias, Exponent};
//! use rand::SeedableRng;
//! use rand::rngs::StdRng;
//!
//! let mut rng = StdRng::seed_from_u64(42);
//!
//! // Generate random bias in range [0, 1)
//! let bias = Bias::rand(&mut rng);
//! assert!(bias >= 0.0 && bias < 1.0);
//!
//! // Generate random exponent (either 0 or 1)
//! let exponent = Exponent::rand(&mut rng);
//! assert!(exponent == 0 || exponent == 1);
//! ```

use rand::Rng;

/// Represents the bias term in a polynomial neuron's activation function.
///
/// The bias is added to the weighted sum of inputs to shift the activation
/// function's output. This allows neurons to have non-zero outputs even when
/// all inputs are zero.
///
/// # Example
///
/// ```
/// use polynomial_neat::core::activation::Bias;
/// use rand::thread_rng;
///
/// let bias_value = Bias::rand(&mut thread_rng());
/// println!("Generated bias: {}", bias_value);
/// ```
pub struct Bias;

impl Bias {
    /// Generates a random bias value in the range [0, 1).
    ///
    /// This method is used during network initialization and mutation to create
    /// diverse bias values that help explore the solution space.
    ///
    /// # Arguments
    ///
    /// * `rng` - A mutable reference to a random number generator
    ///
    /// # Returns
    ///
    /// A random f32 value in the range [0, 1)
    ///
    /// # Example
    ///
    /// ```
    /// use polynomial_neat::core::activation::Bias;
    /// use rand::SeedableRng;
    /// use rand::rngs::StdRng;
    ///
    /// let mut rng = StdRng::seed_from_u64(12345);
    /// let bias = Bias::rand(&mut rng);
    /// assert!(bias >= 0.0 && bias < 1.0);
    /// ```
    pub fn rand(rng: &mut impl Rng) -> f32 {
        rng.random()
    }
}

/// Represents the exponent applied to inputs in a polynomial activation function.
///
/// The exponent determines how each input value is transformed before being
/// weighted and summed. An exponent of 0 makes the input constant (1.0),
/// while an exponent of 1 keeps the input linear.
///
/// # Example
///
/// ```
/// use polynomial_neat::core::activation::Exponent;
/// use rand::thread_rng;
///
/// let exponent = Exponent::rand(&mut thread_rng());
/// match exponent {
///     0 => println!("Input will be raised to power 0 (constant 1.0)"),
///     1 => println!("Input will be raised to power 1 (linear)"),
///     _ => unreachable!(),
/// }
/// ```
pub struct Exponent;

impl Exponent {
    /// Generates a random exponent value (either 0 or 1).
    ///
    /// Currently limited to binary values for simplicity:
    /// - 0: Makes the input constant (x^0 = 1)
    /// - 1: Keeps the input linear (x^1 = x)
    ///
    /// This constraint helps maintain numerical stability while still allowing
    /// for diverse network behaviors.
    ///
    /// # Arguments
    ///
    /// * `rng` - A mutable reference to a random number generator
    ///
    /// # Returns
    ///
    /// An i32 value that is either 0 or 1
    ///
    /// # Example
    ///
    /// ```
    /// use polynomial_neat::core::activation::Exponent;
    /// use rand::SeedableRng;
    /// use rand::rngs::StdRng;
    ///
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let exponent = Exponent::rand(&mut rng);
    /// assert!(exponent == 0 || exponent == 1);
    /// ```
    pub fn rand(rng: &mut impl Rng) -> i32 {
        rng.random_range(0..=1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_bias_range() {
        let mut rng = StdRng::seed_from_u64(42);

        // Test that bias values are always in [0, 1)
        for _ in 0..1000 {
            let bias = Bias::rand(&mut rng);
            assert!(bias >= 0.0, "Bias {} should be >= 0.0", bias);
            assert!(bias < 1.0, "Bias {} should be < 1.0", bias);
        }
    }

    #[test]
    fn test_bias_distribution() {
        let mut rng = StdRng::seed_from_u64(12345);
        let num_samples = 10000;
        let mut sum = 0.0;

        for _ in 0..num_samples {
            sum += Bias::rand(&mut rng);
        }

        let mean = sum / num_samples as f32;

        // For uniform distribution [0, 1), expected mean is 0.5
        // Allow some variance
        assert!(
            (mean - 0.5).abs() < 0.02,
            "Mean bias {} should be close to 0.5",
            mean
        );
    }

    #[test]
    fn test_exponent_values() {
        let mut rng = StdRng::seed_from_u64(9876);

        // Test that exponents are only 0 or 1
        for _ in 0..100 {
            let exp = Exponent::rand(&mut rng);
            assert!(
                exp == 0 || exp == 1,
                "Exponent {} should be either 0 or 1",
                exp
            );
        }
    }

    #[test]
    fn test_exponent_distribution() {
        let mut rng = StdRng::seed_from_u64(5555);
        let num_samples = 1000;
        let mut zeros = 0;
        let mut ones = 0;

        for _ in 0..num_samples {
            match Exponent::rand(&mut rng) {
                0 => zeros += 1,
                1 => ones += 1,
                _ => panic!("Unexpected exponent value"),
            }
        }

        // Check that distribution is roughly 50/50
        let zero_ratio = zeros as f32 / num_samples as f32;
        let one_ratio = ones as f32 / num_samples as f32;

        assert!(
            (zero_ratio - 0.5).abs() < 0.05,
            "Ratio of zeros {} should be close to 0.5",
            zero_ratio
        );
        assert!(
            (one_ratio - 0.5).abs() < 0.05,
            "Ratio of ones {} should be close to 0.5",
            one_ratio
        );
    }

    #[test]
    fn test_deterministic_with_seed() {
        // Test that using the same seed produces the same results
        let seed = 7777;

        let mut rng1 = StdRng::seed_from_u64(seed);
        let mut rng2 = StdRng::seed_from_u64(seed);

        for _ in 0..10 {
            let bias1 = Bias::rand(&mut rng1);
            let bias2 = Bias::rand(&mut rng2);
            assert_eq!(
                bias1, bias2,
                "Bias values should be identical with same seed"
            );

            let exp1 = Exponent::rand(&mut rng1);
            let exp2 = Exponent::rand(&mut rng2);
            assert_eq!(
                exp1, exp2,
                "Exponent values should be identical with same seed"
            );
        }
    }

    #[test]
    fn test_polynomial_activation_example() {
        // Example test showing how bias and exponent work together
        let mut rng = StdRng::seed_from_u64(1111);

        // Simulate a simple polynomial activation
        let input = 2.0_f32;
        let weight = 0.5_f32;
        let exponent = Exponent::rand(&mut rng);
        let bias = Bias::rand(&mut rng);

        let output = weight * input.powi(exponent) + bias;

        match exponent {
            0 => {
                // x^0 = 1, so output = weight * 1 + bias = weight + bias
                let expected = weight + bias;
                assert!(
                    (output - expected).abs() < f32::EPSILON,
                    "Output {} should equal {}",
                    output,
                    expected
                );
            }
            1 => {
                // x^1 = x, so output = weight * x + bias
                let expected = weight * input + bias;
                assert!(
                    (output - expected).abs() < f32::EPSILON,
                    "Output {} should equal {}",
                    output,
                    expected
                );
            }
            _ => panic!("Unexpected exponent"),
        }
    }
}
