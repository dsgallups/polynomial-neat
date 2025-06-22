//! Core neuron implementation for polynomial neural networks.
//!
//! This module provides the fundamental neuron structure used throughout the polynomial
//! NEAT implementation. Neurons in polynomial networks can be of three types:
//!
//! - **Input neurons**: Receive external inputs and have no incoming connections
//! - **Hidden neurons**: Process intermediate computations with polynomial activation
//! - **Output neurons**: Produce the final network outputs
//!
//! # Architecture
//!
//! Each neuron contains:
//! - An inner representation (typically containing the neuron's ID and bias)
//! - Optional properties that define its type and input connections
//!
//! The polynomial activation function computes:
//! ```text
//! output = Σ(weight_i * input_i^exponent_i) + bias
//! ```
//!
//! # Example
//!
//! ```
//! use polynomial_neat::core::neuron::PolyNeuronInner;
//! use polynomial_neat::core::neuron_type::{PolyProps, PropsType};
//! use polynomial_neat::core::input::PolyInput;
//!
//! // Create an input neuron (no properties needed)
//! let input_neuron = PolyNeuronInner::new(1, None);
//! assert!(input_neuron.is_input());
//!
//! // Create a hidden neuron with connections
//! let inputs = vec![PolyInput::new(0, 0.5, 1)];
//! let props = PolyProps::hidden(inputs);
//! let hidden_neuron = PolyNeuronInner::new(2, Some(props));
//! assert!(hidden_neuron.is_hidden());
//! ```

use crate::prelude::*;

/// Core neuron structure for polynomial neural networks.
///
/// This struct encapsulates a neuron's inner representation (typically containing
/// ID and bias) along with optional properties that define its type and connections.
///
/// # Type Parameters
///
/// * `N` - The type of the inner neuron representation
/// * `I` - The type used to identify input connections (typically neuron IDs)
///
/// # Example
///
/// ```
/// use polynomial_neat::core::neuron::PolyNeuronInner;
/// use polynomial_neat::core::neuron_type::PolyProps;
/// use polynomial_neat::core::input::PolyInput;
///
/// // Simple neuron with integer ID
/// let neuron = PolyNeuronInner::new(42, None);
/// assert!(neuron.is_input());
///
/// // Neuron with connections
/// let inputs = vec![PolyInput::new(1, 0.7, 2)];
/// let props = PolyProps::output(inputs);
/// let output_neuron = PolyNeuronInner::new(43, Some(props));
/// assert!(output_neuron.is_output());
/// ```
pub struct PolyNeuronInner<N, I> {
    pub(crate) inner: N,
    props: Option<PolyProps<I>>,
}

impl<N, I> PolyNeuronInner<N, I> {
    /// Creates a new neuron with the given inner representation and optional properties.
    ///
    /// # Arguments
    ///
    /// * `inner` - The inner neuron representation (typically contains ID and bias)
    /// * `props` - Optional properties defining the neuron type and connections
    ///
    /// # Example
    ///
    /// ```
    /// use polynomial_neat::core::neuron::PolyNeuronInner;
    /// use polynomial_neat::core::neuron_type::PolyProps;
    ///
    /// // Input neuron (no properties)
    /// let input = PolyNeuronInner::new("input-1", None);
    ///
    /// // Hidden neuron with empty connections
    /// let hidden_props = PolyProps::hidden(vec![]);
    /// let hidden = PolyNeuronInner::new("hidden-1", Some(hidden_props));
    /// ```
    pub fn new(inner: N, props: Option<PolyProps<I>>) -> Self {
        Self { inner, props }
    }

    /// Returns the neuron's input connections if it has any.
    ///
    /// Input neurons return `None` as they have no incoming connections.
    /// Hidden and output neurons return `Some` with their connections.
    ///
    /// # Returns
    ///
    /// - `None` if this is an input neuron
    /// - `Some(&[PolyInput<I>])` containing the input connections otherwise
    ///
    /// # Example
    ///
    /// ```
    /// use polynomial_neat::core::neuron::PolyNeuronInner;
    /// use polynomial_neat::core::neuron_type::PolyProps;
    /// use polynomial_neat::core::input::PolyInput;
    ///
    /// // Input neuron has no inputs
    /// let input = PolyNeuronInner::new(1, None);
    /// assert!(input.inputs().is_none());
    ///
    /// // Hidden neuron has inputs
    /// let connections = vec![PolyInput::new(0, 0.5, 1)];
    /// let props = PolyProps::hidden(connections);
    /// let hidden = PolyNeuronInner::new(2, Some(props));
    /// assert_eq!(hidden.inputs().unwrap().len(), 1);
    /// ```
    pub fn inputs(&self) -> Option<&[PolyInput<I>]> {
        self.props.as_ref().map(|props| props.inputs())
    }

    /// Returns a reference to the neuron's properties if it has any.
    ///
    /// # Returns
    ///
    /// - `None` for input neurons
    /// - `Some(&PolyProps<I>)` for hidden and output neurons
    ///
    /// # Example
    ///
    /// ```
    /// use polynomial_neat::core::neuron::PolyNeuronInner;
    /// use polynomial_neat::core::neuron_type::PolyProps;
    ///
    /// let input = PolyNeuronInner::new(1, None);
    /// assert!(input.props().is_none());
    ///
    /// let props = PolyProps::hidden(vec![]);
    /// let hidden = PolyNeuronInner::new(2, Some(props.clone()));
    /// assert!(hidden.props().is_some());
    /// ```
    pub fn props(&self) -> Option<&PolyProps<I>> {
        self.props.as_ref()
    }

    /// Returns the type of this neuron.
    ///
    /// The type is determined by the presence and content of properties:
    /// - No properties → Input neuron
    /// - Properties with hidden type → Hidden neuron
    /// - Properties with output type → Output neuron
    ///
    /// # Example
    ///
    /// ```
    /// use polynomial_neat::core::neuron::PolyNeuronInner;
    /// use polynomial_neat::core::neuron_type::{PolyProps, NeuronType};
    ///
    /// let input = PolyNeuronInner::new(1, None);
    /// assert_eq!(input.neuron_type(), NeuronType::input());
    ///
    /// let hidden = PolyNeuronInner::new(2, Some(PolyProps::hidden(vec![])));
    /// assert_eq!(hidden.neuron_type(), NeuronType::hidden());
    ///
    /// let output = PolyNeuronInner::new(3, Some(PolyProps::output(vec![])));
    /// assert_eq!(output.neuron_type(), NeuronType::output());
    /// ```
    pub fn neuron_type(&self) -> NeuronType {
        match self.props {
            None => NeuronType::input(),
            Some(ref props) => props.props_type().into(),
        }
    }

    /// Returns a reference to the inner neuron representation.
    ///
    /// This is typically used internally to access the neuron's ID and bias.
    #[allow(dead_code)]
    pub(crate) fn inner(&self) -> &N {
        &self.inner
    }

    /// Checks if this is an input neuron.
    ///
    /// # Example
    ///
    /// ```
    /// use polynomial_neat::core::neuron::PolyNeuronInner;
    ///
    /// let input = PolyNeuronInner::new(1, None);
    /// assert!(input.is_input());
    /// assert!(!input.is_hidden());
    /// assert!(!input.is_output());
    /// ```
    pub fn is_input(&self) -> bool {
        self.neuron_type() == NeuronType::input()
    }

    /// Checks if this is a hidden neuron.
    ///
    /// # Example
    ///
    /// ```
    /// use polynomial_neat::core::neuron::PolyNeuronInner;
    /// use polynomial_neat::core::neuron_type::PolyProps;
    ///
    /// let hidden = PolyNeuronInner::new(2, Some(PolyProps::hidden(vec![])));
    /// assert!(!hidden.is_input());
    /// assert!(hidden.is_hidden());
    /// assert!(!hidden.is_output());
    /// ```
    pub fn is_hidden(&self) -> bool {
        self.neuron_type() == NeuronType::hidden()
    }

    /// Checks if this is an output neuron.
    ///
    /// # Example
    ///
    /// ```
    /// use polynomial_neat::core::neuron::PolyNeuronInner;
    /// use polynomial_neat::core::neuron_type::PolyProps;
    ///
    /// let output = PolyNeuronInner::new(3, Some(PolyProps::output(vec![])));
    /// assert!(!output.is_input());
    /// assert!(!output.is_hidden());
    /// assert!(output.is_output());
    /// ```
    pub fn is_output(&self) -> bool {
        self.neuron_type() == NeuronType::output()
    }
}

/// Trait for types that behave as neurons in a polynomial network.
///
/// This trait provides a common interface for accessing neuron properties
/// regardless of the specific implementation. It's designed to work with
/// types that wrap or contain a `PolyNeuronInner`.
///
/// # Type Parameters
///
/// * `'a` - Lifetime parameter for returned references
/// * `N` - The type of the inner neuron representation
/// * `I` - The type used to identify input connections
///
/// # Required Methods
///
/// Implementors must provide the `inner()` method to access the underlying
/// `PolyNeuronInner`. All other methods have default implementations that
/// delegate to the inner neuron.
///
/// # Example Implementation
///
/// ```
/// use polynomial_neat::core::neuron::{PolyNeuronInner, Neuron};
///
/// struct MyNeuron {
///     inner: PolyNeuronInner<u32, u32>,
/// }
///
/// impl<'a> Neuron<'a, u32, u32> for MyNeuron {
///     fn inner(&self) -> &PolyNeuronInner<u32, u32> {
///         &self.inner
///     }
/// }
/// ```
pub trait Neuron<'a, N, I>
where
    N: 'a,
    I: 'a,
{
    /// Returns a reference to the inner neuron structure.
    fn inner(&self) -> &PolyNeuronInner<N, I>;

    /// Returns the neuron's input connections if any.
    ///
    /// See [`PolyNeuronInner::inputs`] for details.
    fn inputs(&'a self) -> Option<&'a [PolyInput<I>]> {
        self.inner().inputs()
    }

    /// Returns the neuron's properties if any.
    ///
    /// See [`PolyNeuronInner::props`] for details.
    fn props(&'a self) -> Option<&'a PolyProps<I>> {
        self.inner().props()
    }

    /// Returns the type of this neuron.
    ///
    /// See [`PolyNeuronInner::neuron_type`] for details.
    fn neuron_type(&'a self) -> NeuronType {
        self.inner().neuron_type()
    }

    /// Checks if this is an input neuron.
    ///
    /// See [`PolyNeuronInner::is_input`] for details.
    fn is_input(&'a self) -> bool {
        self.inner().is_input()
    }

    /// Checks if this is a hidden neuron.
    ///
    /// See [`PolyNeuronInner::is_hidden`] for details.
    fn is_hidden(&'a self) -> bool {
        self.inner().is_hidden()
    }

    /// Checks if this is an output neuron.
    ///
    /// See [`PolyNeuronInner::is_output`] for details.
    fn is_output(&'a self) -> bool {
        self.inner().is_output()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test implementation of the Neuron trait
    struct TestNeuron {
        inner: PolyNeuronInner<u32, u32>,
    }

    impl<'a> Neuron<'a, u32, u32> for TestNeuron {
        fn inner(&self) -> &PolyNeuronInner<u32, u32> {
            &self.inner
        }
    }

    #[test]
    fn test_input_neuron_creation() {
        let neuron: PolyNeuronInner<_, u32> = PolyNeuronInner::new(42u32, None);

        assert!(neuron.is_input());
        assert!(!neuron.is_hidden());
        assert!(!neuron.is_output());
        assert_eq!(neuron.neuron_type(), NeuronType::input());
        assert!(neuron.inputs().is_none());
        assert!(neuron.props().is_none());
    }

    #[test]
    fn test_hidden_neuron_creation() {
        let inputs = vec![PolyInput::new(1u32, 0.5, 1), PolyInput::new(2u32, -0.3, 2)];
        let props = PolyProps::hidden(inputs);
        let neuron = PolyNeuronInner::new(43u32, Some(props));

        assert!(!neuron.is_input());
        assert!(neuron.is_hidden());
        assert!(!neuron.is_output());
        assert_eq!(neuron.neuron_type(), NeuronType::hidden());

        let neuron_inputs = neuron.inputs().unwrap();
        assert_eq!(neuron_inputs.len(), 2);
        assert_eq!(*neuron_inputs[0].input(), 1u32);
        assert_eq!(neuron_inputs[0].weight(), 0.5);
        assert_eq!(neuron_inputs[1].weight(), -0.3);
    }

    #[test]
    fn test_output_neuron_creation() {
        let inputs = vec![PolyInput::new(3u32, 0.8, 0)];
        let props = PolyProps::output(inputs);
        let neuron = PolyNeuronInner::new(44u32, Some(props));

        assert!(!neuron.is_input());
        assert!(!neuron.is_hidden());
        assert!(neuron.is_output());
        assert_eq!(neuron.neuron_type(), NeuronType::output());

        assert!(neuron.inputs().is_some());
        assert!(neuron.props().is_some());
    }

    #[test]
    fn test_neuron_with_no_connections() {
        let props = PolyProps::hidden(vec![]);
        let neuron: PolyNeuronInner<_, i32> = PolyNeuronInner::new(45u32, Some(props));

        assert!(neuron.is_hidden());
        let inputs = neuron.inputs().unwrap();
        assert_eq!(inputs.len(), 0);
    }

    #[test]
    fn test_trait_implementation() {
        let inner = PolyNeuronInner::new(100u32, None);
        let test_neuron = TestNeuron { inner };

        assert!(test_neuron.is_input());
        assert!(!test_neuron.is_hidden());
        assert!(!test_neuron.is_output());
        assert_eq!(test_neuron.neuron_type(), NeuronType::input());
        assert!(test_neuron.inputs().is_none());
        assert!(test_neuron.props().is_none());
    }

    #[test]
    fn test_trait_with_connections() {
        let inputs = vec![
            PolyInput::new(10u32, 0.1, 1),
            PolyInput::new(20u32, 0.2, 0),
            PolyInput::new(30u32, 0.3, 2),
        ];
        let props = PolyProps::output(inputs);
        let inner = PolyNeuronInner::new(101u32, Some(props));
        let test_neuron = TestNeuron { inner };

        assert!(test_neuron.is_output());
        let neuron_inputs = test_neuron.inputs().unwrap();
        assert_eq!(neuron_inputs.len(), 3);
        assert_eq!(*neuron_inputs[0].input(), 10u32);
        assert_eq!(*neuron_inputs[1].input(), 20u32);
        assert_eq!(*neuron_inputs[2].input(), 30u32);
    }

    #[test]
    fn test_inner_access() {
        let neuron: PolyNeuronInner<_, f32> = PolyNeuronInner::new(55u32, None);
        assert_eq!(*neuron.inner(), 55u32);
    }

    #[test]
    fn test_different_inner_types() {
        // Test with string inner type
        let string_neuron: PolyNeuronInner<_, f32> =
            PolyNeuronInner::new("neuron-a".to_string(), None);
        assert!(string_neuron.is_input());
        assert_eq!(*string_neuron.inner(), "neuron-a".to_string());

        // Test with tuple inner type
        let tuple_neuron: PolyNeuronInner<_, f32> = PolyNeuronInner::new((1, 0.5f32), None);
        assert!(tuple_neuron.is_input());
        assert_eq!(*tuple_neuron.inner(), (1, 0.5f32));
    }

    #[test]
    fn test_props_access() {
        let inputs = vec![PolyInput::new(1u32, 0.5, 1)];
        let props = PolyProps::hidden(inputs.clone());
        let neuron = PolyNeuronInner::new(60u32, Some(props));

        let neuron_props = neuron.props().unwrap();
        assert_eq!(neuron_props.props_type(), PropsType::Hidden);
        assert_eq!(neuron_props.num_inputs(), 1);
    }
}
