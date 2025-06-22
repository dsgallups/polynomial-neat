//! Input connections for polynomial neural networks.
//!
//! This module defines the [`PolyInput`] struct, which represents a weighted connection
//! from one neuron to another in a polynomial neural network. Each connection has:
//!
//! - An input reference (typically a neuron ID)
//! - A weight that scales the input value
//! - An exponent that transforms the input value
//!
//! The contribution of each input to a neuron's activation is calculated as:
//! ```text
//! contribution = weight * input_value^exponent
//! ```
//!
//! # Example
//!
//! ```
//! use polynomial_neat::core::input::PolyInput;
//! use rand::SeedableRng;
//! use rand::rngs::StdRng;
//!
//! // Create a connection from neuron ID 5 with specific parameters
//! let input = PolyInput::new(5, 0.7, 2);
//! assert_eq!(*input.input(), 5);
//! assert_eq!(input.weight(), 0.7);
//! assert_eq!(input.exponent(), 2);
//!
//! // Create a connection with random parameters
//! let mut rng = StdRng::seed_from_u64(42);
//! let random_input = PolyInput::new_rand(10, &mut rng);
//! assert_eq!(*random_input.input(), 10);
//! assert!(random_input.weight() >= -1.0 && random_input.weight() <= 1.0);
//! assert!(random_input.exponent() >= 0 && random_input.exponent() <= 2);
//! ```

use rand::Rng;

/// Represents a weighted input connection in a polynomial neural network.
///
/// Each `PolyInput` encapsulates:
/// - The source of the input (typically a neuron identifier)
/// - The connection weight
/// - The exponent applied to the input value
///
/// The generic type `I` represents the input identifier type, which is typically
/// a neuron ID but can be any type that identifies the source of the input.
///
/// # Type Parameters
///
/// * `I` - The type used to identify the input source (e.g., neuron ID)
///
/// # Example
///
/// ```
/// use polynomial_neat::core::input::PolyInput;
///
/// // Using neuron IDs as integers
/// let input1 = PolyInput::new(42, 0.5, 1);
///
/// // Using neuron IDs as UUIDs (example with String for simplicity)
/// let input2 = PolyInput::new("neuron-123".to_string(), -0.3, 0);
/// ```
#[derive(Clone, Debug)]
pub struct PolyInput<I> {
    input: I,
    weight: f32,
    exp: i32,
}

impl<I> PolyInput<I> {
    /// Creates a new `PolyInput` with specified parameters.
    ///
    /// # Arguments
    ///
    /// * `input` - The identifier of the input source
    /// * `weight` - The connection weight (can be any float value)
    /// * `exp` - The exponent applied to the input value
    ///
    /// # Example
    ///
    /// ```
    /// use polynomial_neat::core::input::PolyInput;
    ///
    /// let input = PolyInput::new(5, -0.8, 2);
    /// assert_eq!(*input.input(), 5);
    /// assert_eq!(input.weight(), -0.8);
    /// assert_eq!(input.exponent(), 2);
    /// ```
    pub fn new(input: I, weight: f32, exp: i32) -> Self {
        Self { input, weight, exp }
    }

    /// Creates a new `PolyInput` with random weight and exponent.
    ///
    /// The random values are generated within specific ranges:
    /// - Weight: [-1.0, 1.0]
    /// - Exponent: [0, 2] (inclusive)
    ///
    /// These ranges are chosen to provide good initial diversity while
    /// maintaining numerical stability.
    ///
    /// # Arguments
    ///
    /// * `input` - The identifier of the input source
    /// * `rng` - A mutable reference to a random number generator
    ///
    /// # Example
    ///
    /// ```
    /// use polynomial_neat::core::input::PolyInput;
    /// use rand::SeedableRng;
    /// use rand::rngs::StdRng;
    ///
    /// let mut rng = StdRng::seed_from_u64(12345);
    /// let input = PolyInput::new_rand(7, &mut rng);
    ///
    /// assert_eq!(*input.input(), 7);
    /// assert!(input.weight() >= -1.0 && input.weight() <= 1.0);
    /// assert!(input.exponent() >= 0 && input.exponent() <= 2);
    /// ```
    pub fn new_rand(input: I, rng: &mut impl Rng) -> Self {
        Self {
            input,
            weight: rng.random_range(-1.0..=1.0),
            exp: rng.random_range(0..=2),
        }
    }

    /// Returns a reference to the input identifier.
    ///
    /// # Example
    ///
    /// ```
    /// use polynomial_neat::core::input::PolyInput;
    ///
    /// let input = PolyInput::new("neuron-a", 0.5, 1);
    /// assert_eq!(input.input(), &"neuron-a");
    /// ```
    pub fn input(&self) -> &I {
        &self.input
    }

    /// Returns the connection weight.
    ///
    /// # Example
    ///
    /// ```
    /// use polynomial_neat::core::input::PolyInput;
    ///
    /// let input = PolyInput::new(1, 0.75, 2);
    /// assert_eq!(input.weight(), 0.75);
    /// ```
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Adjusts the connection weight by adding the specified delta.
    ///
    /// This method is typically used during mutation to fine-tune weights.
    ///
    /// # Arguments
    ///
    /// * `by` - The amount to add to the current weight (can be negative)
    ///
    /// # Example
    ///
    /// ```
    /// use polynomial_neat::core::input::PolyInput;
    ///
    /// let mut input = PolyInput::new(1, 0.5, 1);
    /// input.adjust_weight(0.2);
    /// assert_eq!(input.weight(), 0.7);
    ///
    /// input.adjust_weight(-0.3);
    ///
    /// assert!((input.weight() - 0.4).abs() < std::f32::EPSILON);
    /// ```
    pub fn adjust_weight(&mut self, by: f32) {
        self.weight += by;
    }

    /// Returns the exponent applied to the input value.
    ///
    /// # Example
    ///
    /// ```
    /// use polynomial_neat::core::input::PolyInput;
    ///
    /// let input = PolyInput::new(1, 0.5, 3);
    /// assert_eq!(input.exponent(), 3);
    /// ```
    pub fn exponent(&self) -> i32 {
        self.exp
    }

    /// Adjusts the exponent by adding the specified delta.
    ///
    /// This method is typically used during mutation to modify the polynomial
    /// behavior of the connection.
    ///
    /// # Arguments
    ///
    /// * `by` - The amount to add to the current exponent (can be negative)
    ///
    /// # Example
    ///
    /// ```
    /// use polynomial_neat::core::input::PolyInput;
    ///
    /// let mut input = PolyInput::new(1, 0.5, 1);
    /// input.adjust_exp(1);
    /// assert_eq!(input.exponent(), 2);
    ///
    /// input.adjust_exp(-2);
    /// assert_eq!(input.exponent(), 0);
    /// ```
    pub fn adjust_exp(&mut self, by: i32) {
        self.exp += by;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_new() {
        let input = PolyInput::new(42, 0.5, 2);
        assert_eq!(*input.input(), 42);
        assert_eq!(input.weight(), 0.5);
        assert_eq!(input.exponent(), 2);
    }

    #[test]
    fn test_new_with_negative_weight() {
        let input = PolyInput::new("test", -0.75, 0);
        assert_eq!(*input.input(), "test");
        assert_eq!(input.weight(), -0.75);
        assert_eq!(input.exponent(), 0);
    }

    #[test]
    fn test_new_rand_ranges() {
        let mut rng = StdRng::seed_from_u64(42);

        // Test multiple random generations to ensure ranges are respected
        for _ in 0..100 {
            let input = PolyInput::new_rand(1, &mut rng);
            assert!(
                input.weight() >= -1.0 && input.weight() <= 1.0,
                "Weight {} should be in range [-1.0, 1.0]",
                input.weight()
            );
            assert!(
                input.exponent() >= 0 && input.exponent() <= 2,
                "Exponent {} should be in range [0, 2]",
                input.exponent()
            );
        }
    }

    #[test]
    fn test_new_rand_distribution() {
        let mut rng = StdRng::seed_from_u64(12345);
        let num_samples = 1000;

        let mut weight_sum = 0.0;
        let mut exp_counts = [0; 3]; // For exponents 0, 1, 2

        for _ in 0..num_samples {
            let input = PolyInput::new_rand(1, &mut rng);
            weight_sum += input.weight();
            exp_counts[input.exponent() as usize] += 1;
        }

        // Check weight distribution (should average near 0)
        let weight_mean = weight_sum / num_samples as f32;
        assert!(
            weight_mean.abs() < 0.1,
            "Weight mean {} should be close to 0",
            weight_mean
        );

        // Check exponent distribution (should be roughly uniform)
        for (exp, count) in exp_counts.iter().enumerate() {
            let ratio = *count as f32 / num_samples as f32;
            assert!(
                (ratio - 0.333).abs() < 0.05,
                "Exponent {} ratio {} should be close to 0.333",
                exp,
                ratio
            );
        }
    }

    #[test]
    fn test_adjust_weight() {
        let mut input = PolyInput::new(1, 0.5, 1);

        input.adjust_weight(0.3);
        assert!((input.weight() - 0.8).abs() < f32::EPSILON);

        input.adjust_weight(-0.5);
        assert!((input.weight() - 0.3).abs() < f32::EPSILON);

        input.adjust_weight(-0.3);
        assert!((input.weight() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_adjust_exp() {
        let mut input = PolyInput::new(1, 0.5, 1);

        input.adjust_exp(2);
        assert_eq!(input.exponent(), 3);

        input.adjust_exp(-1);
        assert_eq!(input.exponent(), 2);

        input.adjust_exp(-2);
        assert_eq!(input.exponent(), 0);

        // Test negative exponents
        input.adjust_exp(-1);
        assert_eq!(input.exponent(), -1);
    }

    #[test]
    fn test_clone() {
        let original = PolyInput::new(42, 0.7, 2);
        let cloned = original.clone();

        assert_eq!(*cloned.input(), *original.input());
        assert_eq!(cloned.weight(), original.weight());
        assert_eq!(cloned.exponent(), original.exponent());
    }

    #[test]
    fn test_debug_format() {
        let input = PolyInput::new(123, 0.5, 1);
        let debug_str = format!("{:?}", input);

        assert!(debug_str.contains("PolyInput"));
        assert!(debug_str.contains("123"));
        assert!(debug_str.contains("0.5"));
        assert!(debug_str.contains("1"));
    }

    #[test]
    fn test_deterministic_rand() {
        let seed = 9876;
        let mut rng1 = StdRng::seed_from_u64(seed);
        let mut rng2 = StdRng::seed_from_u64(seed);

        for i in 0..10 {
            let input1 = PolyInput::new_rand(i, &mut rng1);
            let input2 = PolyInput::new_rand(i, &mut rng2);

            assert_eq!(input1.weight(), input2.weight());
            assert_eq!(input1.exponent(), input2.exponent());
        }
    }

    #[test]
    fn test_polynomial_calculation_example() {
        // Example showing how the polynomial input would be used
        let input = PolyInput::new(1, 0.5, 2);
        let input_value = 3.0_f32;

        // Calculate contribution: weight * input_value^exponent
        let contribution = input.weight() * input_value.powi(input.exponent());
        let expected = 0.5 * 3.0_f32.powi(2); // 0.5 * 9 = 4.5

        assert!((contribution - expected).abs() < f32::EPSILON);
    }

    #[test]
    fn test_with_different_input_types() {
        // Test with string IDs
        let string_input = PolyInput::new("neuron-a", 0.5, 1);
        assert_eq!(*string_input.input(), "neuron-a");

        // Test with usize IDs
        let usize_input = PolyInput::new(42usize, 0.5, 1);
        assert_eq!(*usize_input.input(), 42usize);

        // Test with custom type
        #[derive(Debug, Clone, PartialEq)]
        struct NeuronId(u64);

        let custom_input = PolyInput::new(NeuronId(123), 0.5, 1);
        assert_eq!(*custom_input.input(), NeuronId(123));
    }
}
