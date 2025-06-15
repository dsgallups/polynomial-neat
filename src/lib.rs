//! # burn-neat
//!
//! A Rust implementation of NEAT (NeuroEvolution of Augmenting Topologies) using the Burn deep learning framework.
//!
//! This crate provides a flexible and performant implementation of the NEAT algorithm, which evolves neural network
//! topologies and weights through genetic algorithms. It supports both CPU and GPU computation through Burn's
//! backend system.
//!
//! ## Overview
//!
//! NEAT is a genetic algorithm for evolving artificial neural networks. It starts with simple networks and
//! complexifies them over generations by adding nodes and connections through mutations. This implementation
//! provides:
//!
//! - **Polynomial Networks**: Networks that use polynomial activation functions with learnable exponents
//! - **Topology Evolution**: Automatic addition/removal of neurons and connections
//! - **Speciation**: Grouping of similar network topologies (coming soon)
//! - **GPU Acceleration**: Via Burn's CUDA and WGPU backends
//!
//! ## Quick Start
//!
//! ```rust
//! use burn_neat::poly::prelude::*;
//! use burn_neat::poly::topology::mutation::MutationChances;
//!
//! // Configure mutation chances for evolution
//! let mutation_chances = MutationChances::new_from_raw(
//!     3,      // max mutations per generation
//!     80.0,   // add neuron chance
//!     50.0,   // add connection chance
//!     5.0,    // remove neuron chance
//!     60.0,   // mutate weight chance
//!     20.0    // mutate exponent chance
//! );
//!
//! // Create a network topology with 2 inputs and 2 outputs
//! let mut topology = PolyNetworkTopology::new(
//!     2,
//!     2,
//!     mutation_chances,
//!     &mut rand::rng()
//! );
//!
//! // Evolve the topology
//! topology = topology.replicate(&mut rand::rng());
//!
//! // Convert to a runnable network
//! let network = topology.to_simple_network();
//!
//! // Make predictions
//! let inputs = vec![1.0, 0.5];
//! let outputs: Vec<f32> = network.predict(&inputs).collect();
//!
//! println!("Network output: {:?}", outputs);
//! ```
//!
//! ## Creating Networks
//!
//! There are several ways to create networks:
//!
//! ### Random Initialization
//! ```rust
//! # use burn_neat::poly::prelude::*;
//! # use burn_neat::poly::topology::mutation::MutationChances;
//! # let mutation_chances = MutationChances::new(50);
//! // Creates a randomly connected network
//! let topology = PolyNetworkTopology::new(3, 1, mutation_chances, &mut rand::rng());
//! ```
//!
//! ### Fully Connected
//! ```rust
//! # use burn_neat::poly::prelude::*;
//! # use burn_neat::poly::topology::mutation::MutationChances;
//! # let mutation_chances = MutationChances::new(50);
//! // Creates a fully connected network with all inputs connected to all outputs
//! let topology = PolyNetworkTopology::new_thoroughly_connected(
//!     4, 2, mutation_chances, &mut rand::rng()
//! );
//! ```
//!
//! ## Network Evolution
//!
//! Networks evolve through mutations controlled by `MutationChances`:
//!
//! ```rust
//! # use burn_neat::poly::prelude::*;
//! # use burn_neat::poly::topology::mutation::MutationChances;
//! let mutation_chances = MutationChances::new_from_raw(
//!     5,      // max mutations
//!     80.0,   // split connection chance (add neuron)
//!     50.0,   // add connection chance
//!     5.0,    // remove neuron chance
//!     60.0,   // mutate weight chance
//!     20.0    // mutate exponent chance
//! );
//! ```
//!
//! ## GPU Acceleration
//!
//! For GPU acceleration, use the Burn backend networks:
//!
//! ```rust
//! # use burn_neat::poly::prelude::*;
//! # use burn_neat::poly::topology::mutation::MutationChances;
//! use burn_neat::poly::burn_net::network::BurnNetwork;
//! use burn::backend::NdArray;
//!
//! # let mutation_chances = MutationChances::new(50);
//! # let topology = PolyNetworkTopology::new(2, 2, mutation_chances, &mut rand::rng());
//! // Create network on CPU backend
//! let device = burn::backend::ndarray::NdArrayDevice::default();
//! let burn_network = BurnNetwork::<NdArray>::from_topology(&topology, device);
//!
//! // Make predictions
//! let outputs = burn_network.predict(&[1.0, 0.5]);
//! assert_eq!(outputs.len(), 2); // Two output neurons
//! ```
//!
//! ## Module Structure
//!
//! - [`poly`]: Polynomial network implementation with configurable activation functions
//! - [`activated`]: Traditional NEAT implementation with fixed activation functions
//! - [`core`]: Core traits and utilities shared across implementations

/// Traditional NEAT implementation with fixed activation functions.
///
/// This module provides the classic NEAT algorithm implementation with predetermined
/// activation functions like sigmoid, tanh, ReLU, etc.
pub mod activated;

/// Polynomial network implementation with learnable activation functions.
///
/// This module extends NEAT with polynomial activation functions where the exponents
/// can be evolved alongside the network topology. This allows for more flexible
/// function approximation.
pub mod poly;

mod test_utils;

/// Core functionality and traits used throughout the crate.
pub mod core;
