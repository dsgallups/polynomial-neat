//! Polynomial Neural Networks with Evolvable Activation Functions
//!
//! This module implements a novel approach to NEAT where neurons use polynomial activation
//! functions with learnable exponents. Unlike traditional neural networks with fixed
//! activation functions (sigmoid, tanh, ReLU), polynomial networks can evolve the shape
//! of their activation functions during training.
//!
//! ## Key Concepts
//!
//! ### Polynomial Activation
//! Each neuron computes its output as:
//! ```text
//! output = Σ(weight_i * input_i^exponent_i) + bias
//! ```
//! Where both weights and exponents can be evolved.
//!
//! ### Network Structure
//! - **Input Layer**: Neurons that receive external inputs
//! - **Hidden Layer**: Neurons that process intermediate computations
//! - **Output Layer**: Neurons that produce the final outputs
//!
//! ### Evolution Process
//! Networks evolve through mutations:
//! - Adding/removing neurons
//! - Adding/removing connections
//! - Mutating weights
//! - Mutating exponents
//!
//! ## Example
//!
//! ```rust
//! use polynomial_neat::poly::prelude::*;
//! use polynomial_neat::poly::topology::mutation::MutationChances;
//!
//! // Create mutation parameters
//! let mutations = MutationChances::new_from_raw(3, 80.0, 50.0, 5.0, 60.0, 20.0);
//!
//! // Create a network with 3 inputs and 1 output
//! let mut topology = PolyNetworkTopology::new(3, 1, mutations, &mut rand::rng());
//!
//! // Evolve the network
//! for generation in 0..10 {
//!     topology = topology.replicate(&mut rand::rng());
//!
//!     // Convert to executable network
//!     let network = topology.to_simple_network();
//!
//!     // Run inference
//!     let inputs = vec![1.0, 2.0, 3.0];
//!     let outputs: Vec<f32> = network.predict(&inputs).collect();
//!
//!     println!("Generation {}: Output = {:?}", generation, outputs);
//! }
//! ```
//!
//! ## Advanced Example: Solving XOR
//!
//! The XOR problem is a classic benchmark for neural networks. Here's how to solve it
//! using polynomial NEAT:
//!
//! ```rust
//! use polynomial_neat::poly::prelude::*;
//! use polynomial_neat::poly::topology::mutation::MutationChances;
//!
//! // Define fitness function for XOR
//! fn evaluate_xor(network: &SimplePolyNetwork) -> f32 {
//!     let test_cases = [
//!         ([0.0, 0.0], 0.0),
//!         ([0.0, 1.0], 1.0),
//!         ([1.0, 0.0], 1.0),
//!         ([1.0, 1.0], 0.0),
//!     ];
//!
//!     let mut total_error = 0.0;
//!     for (inputs, expected) in test_cases.iter() {
//!         let output: Vec<f32> = network.predict(inputs).collect();
//!         let error = (output[0] - expected).abs();
//!         total_error += error;
//!     }
//!
//!     // Convert error to fitness (higher is better)
//!     4.0 - total_error
//! }
//!
//! // Set up evolution
//! let mutations = MutationChances::new_from_raw(
//!     3,      // max mutations per generation
//!     80.0,   // high chance to add neurons
//!     50.0,   // moderate chance to add connections
//!     5.0,    // low chance to remove neurons
//!     60.0,   // moderate weight mutation
//!     20.0    // low exponent mutation
//! );
//!
//! // Create initial population
//! let mut population: Vec<PolyNetworkTopology> = (0..10)
//!     .map(|_| PolyNetworkTopology::new(2, 1, mutations, &mut rand::rng()))
//!     .collect();
//!
//! // Evolution loop
//! for generation in 0..50 {
//!     // Evaluate fitness for each network
//!     let mut fitness_scores: Vec<(usize, f32)> = population
//!         .iter()
//!         .enumerate()
//!         .map(|(idx, topology)| {
//!             let network = topology.to_simple_network();
//!             (idx, evaluate_xor(&network))
//!         })
//!         .collect();
//!
//!     // Sort by fitness (best first)
//!     fitness_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
//!
//!     // Check if we found a solution
//!     if fitness_scores[0].1 > 3.95 {
//!         println!("Solution found in generation {}!", generation);
//!         break;
//!     }
//!
//!     // Create next generation (keep best, mutate others)
//!     let best_idx = fitness_scores[0].0;
//!     let best = population[best_idx].deep_clone();
//!
//!     population = (0..10)
//!         .map(|i| {
//!             if i == 0 {
//!                 best.deep_clone() // Keep best unchanged
//!             } else {
//!                 best.replicate(&mut rand::rng()) // Mutate copies of best
//!             }
//!         })
//!         .collect();
//! }
//!
//! // Use the best network
//! let best_network = population[0].to_simple_network();
//! println!("XOR(0,0) = {:?}", best_network.predict(&[0.0, 0.0]).collect::<Vec<_>>());
//! println!("XOR(0,1) = {:?}", best_network.predict(&[0.0, 1.0]).collect::<Vec<_>>());
//! println!("XOR(1,0) = {:?}", best_network.predict(&[1.0, 0.0]).collect::<Vec<_>>());
//! println!("XOR(1,1) = {:?}", best_network.predict(&[1.0, 1.0]).collect::<Vec<_>>());
//! ```
//!
//! ## Submodules
//!
//! - `burn_net`: GPU-accelerated implementation using Burn framework
//! - `core`: Core components like activation functions and neuron types
//! - `simple_net`: CPU-based implementation for testing and development
//! - `topology`: Network topology representation and mutation logic

/// GPU-accelerated polynomial network implementation using Burn.
///
/// This module provides high-performance network execution on CUDA and WGPU devices.
pub mod burn_net;
// pub mod candle_net;  // Commented out - replaced by burn_net

/// Core components for polynomial networks.
///
/// Includes activation functions, neuron implementations, and input handling.
pub mod core;

/// Simple CPU-based polynomial network implementation.
///
/// Useful for debugging, testing, and environments without GPU support.
pub mod simple_net;

/// Network topology representation and evolution.
///
/// Handles the structure of networks and how they mutate over generations.
pub mod topology;
/// Common imports for working with polynomial networks.
///
/// ## Example
/// ```rust
/// use polynomial_neat::poly::prelude::*;
///
/// // Now you have access to all common types like:
/// // - PolyNetworkTopology
/// // - SimplePolyNetwork
/// // - MutationChances
/// // - NeuronType
/// // etc.
/// ```
pub mod prelude {
    pub use super::core::{
        activation::{Bias, Exponent},
        input::PolyInput,
        //neuron::PolyNeuronInner,
        neuron_type::{NeuronType, PolyProps, PropsType},
    };
    pub use super::simple_net::{
        input::NeuronInput, network::SimplePolyNetwork, neuron::SimpleNeuron,
        neuron_type::NeuronProps,
    };
    pub use super::topology::{
        input::PolyInputTopology,
        mutation::{MAX_MUTATIONS, MutationAction, MutationChances},
        network::PolyNetworkTopology,
        neuron::PolyNeuronTopology,
        neuron_type::PolyNeuronPropsTopology,
    };
    #[cfg(test)]
    pub(crate) use crate::test_utils::arc;
}

#[cfg(test)]
mod tests;
