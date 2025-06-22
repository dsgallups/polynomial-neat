# polynomial-neat

A high-performance Rust implementation of NEAT (NeuroEvolution of Augmenting Topologies) using the Burn deep learning framework. This crate provides a novel extension to NEAT with polynomial activation functions, allowing networks to evolve both their topology and activation function shapes.

[![Crates.io](https://img.shields.io/crates/v/polynomial-neat.svg)](https://crates.io/crates/polynomial-neat)
[![Documentation](https://docs.rs/polynomial-neat/badge.svg)](https://docs.rs/polynomial-neat)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

## Features

- **Polynomial Activation Functions**: Unlike traditional NEAT with fixed activation functions, this implementation allows neurons to evolve polynomial activation functions with learnable exponents
- **GPU Acceleration**: Full support for CUDA and WGPU backends through Burn
- **Flexible Topology Evolution**: Automatic addition/removal of neurons and connections
- **Thread-Safe**: Parallel evaluation of networks using Rayon
- **Configurable Mutations**: Fine-grained control over evolution parameters

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
polynomial-neat = "0.1.0"
```

For GPU support, enable the appropriate features:

```toml
[dependencies]
polynomial-neat = { version = "0.1.0", features = ["cuda"] }
# or
polynomial-neat = { version = "0.1.0", features = ["wgpu"] }
```

## Quick Start

```rust
use polynomial_neat::poly::prelude::*;
use polynomial_neat::poly::topology::mutation::MutationChances;

fn main() {
    // Configure mutation parameters
    let mutations = MutationChances::new_from_raw(
        3,      // max mutations per generation
        80.0,   // chance to add neuron (split connection)
        50.0,   // chance to add connection
        5.0,    // chance to remove neuron
        60.0,   // chance to mutate weight
        20.0    // chance to mutate exponent
    );

    // Create a network with 2 inputs and 1 output
    let mut topology = PolyNetworkTopology::new(
        2,
        1,
        mutations,
        &mut rand::rng()
    );

    // Evolve for 10 generations
    for generation in 0..10 {
        // Mutate the topology
        topology = topology.replicate(&mut rand::rng());

        // Convert to executable network
        let network = topology.to_simple_network();

        // Evaluate the network
        let output: Vec<f32> = network.predict(&[1.0, 0.5]).collect();

        println!("Generation {}: Output = {:?}", generation, output);
    }
}
```

## Core Concepts

### Polynomial Neurons

Each neuron in the network computes its output using a polynomial activation function:

```text
(TODO: this needs to be rewritten).
output = ΣkΣi(weight_i_k * input_k^exponent_i_k)
```

This allows the network to learn complex non-linear transformations by evolving both the weights and exponents.

### Network Topology

Networks consist of three types of neurons:
- **Input neurons**: Receive external inputs
- **Hidden neurons**: Process intermediate computations
- **Output neurons**: Produce final outputs

### Evolution Process

Networks evolve through several types of mutations:

1. **Split Connection**: Add a new neuron between two connected neurons
2. **Add Connection**: Create a new connection between neurons
3. **Remove Neuron**: Delete a hidden neuron and its connections
4. **Mutate Weight**: Adjust connection weights
5. **Mutate Exponent**: Modify polynomial exponents

## Detailed Examples

### XOR Problem

```rust
use polynomial_neat::poly::prelude::*;
use polynomial_neat::poly::topology::mutation::MutationChances;

fn evaluate_xor(network: &SimplePolyNetwork) -> f32 {
    let test_cases = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];

    let mut error = 0.0;
    for (inputs, expected) in test_cases.iter() {
        let output: Vec<f32> = network.predict(inputs).collect();
        error += (output[0] - expected).powi(2);
    }

    4.0 - error // Higher is better
}

fn main() {
    let mutations = MutationChances::new_from_raw(3, 80.0, 50.0, 5.0, 60.0, 20.0);
    let mut best_topology = PolyNetworkTopology::new(2, 1, mutations, &mut rand::rng());
    let mut best_fitness = 0.0;

    for generation in 0..100 {
        // Create offspring
        let offspring = best_topology.replicate(&mut rand::rng());
        let network = offspring.to_simple_network();
        let fitness = evaluate_xor(&network);

        // Keep if better
        if fitness > best_fitness {
            best_topology = offspring;
            best_fitness = fitness;
            println!("Generation {}: Fitness = {}", generation, fitness);
        }

        if best_fitness > 3.9 {
            println!("Solution found!");
            break;
        }
    }
}
```

### GPU-Accelerated Networks

```rust
use polynomial_neat::poly::prelude::*;
use polynomial_neat::poly::burn_net::network::BurnNetwork;
use burn::backend::Cuda;

fn main() {
    // Create topology
    let mutations = MutationChances::new(50);
    let topology = PolyNetworkTopology::new_thoroughly_connected(
        4, 2, mutations, &mut rand::rng()
    );

    // Create GPU-accelerated network
    let device = burn::backend::cuda::CudaDevice::default();
    let network = BurnNetwork::<Cuda>::from_topology(&topology, device);

    // Run inference on GPU
    let inputs = vec![1.0, 2.0, 3.0, 4.0];
    let outputs = network.predict(&inputs);

    println!("GPU outputs: {:?}", outputs);
}
```

### Custom Mutation Strategy

```rust
use polynomial_neat::poly::prelude::*;
use polynomial_neat::poly::topology::mutation::MutationChances;

fn main() {
    // Start with aggressive topology changes
    let mut mutations = MutationChances::new_from_raw(
        5,     // Many mutations per step
        90.0,  // Very high chance to add neurons
        80.0,  // High chance to add connections
        1.0,   // Very low chance to remove
        30.0,  // Low weight mutation
        10.0   // Low exponent mutation
    );

    let mut topology = PolyNetworkTopology::new(3, 2, mutations, &mut rand::rng());

    // Evolve with changing strategy
    for generation in 0..100 {
        if generation == 50 {
            // Switch to fine-tuning after 50 generations
            mutations = MutationChances::new_from_raw(
                2,     // Fewer mutations
                10.0,  // Low topology changes
                10.0,
                5.0,
                80.0,  // High weight mutation for fine-tuning
                60.0   // High exponent mutation
            );
            topology = PolyNetworkTopology::from_raw_parts(
                topology.neurons().clone(),
                mutations
            );
        }

        topology = topology.replicate(&mut rand::rng());

        let network = topology.to_simple_network();
        println!("Generation {}: {} neurons", generation, network.num_nodes());
    }
}
```

## Architecture

The crate is organized into two main modules:

- **`poly`**: Polynomial network implementation with evolvable activation functions
- **`activated`**: Traditional NEAT with fixed activation functions (coming soon)

### Key Types

- `PolyNetworkTopology`: Represents network structure and evolution parameters
- `SimplePolyNetwork`: CPU-based network for inference
- `BurnNetwork`: GPU-accelerated network using Burn
- `MutationChances`: Configuration for evolution probabilities
- `PolyNeuronTopology`: Individual neuron representation

## Performance Considerations

- **CPU vs GPU**: Use `SimplePolyNetwork` for small networks or CPU-only environments. Use `BurnNetwork` with CUDA/WGPU for larger networks or batch processing.
- **Mutation Rate**: Higher mutation rates explore more but may be unstable. Start with moderate rates and adjust based on your problem.
- **Network Size**: Larger networks are more expressive but slower. The algorithm starts small and complexifies as needed.

## Roadmap

- [x] Core polynomial NEAT implementation
- [x] CPU-based inference
- [x] GPU acceleration with Burn
- [ ] Speciation for diversity preservation
- [ ] Recurrent connections
- [ ] Traditional activation functions
- [ ] Serialization/deserialization
- [ ] Benchmark suite

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{polynomial-neat,
  author = {Gallups, Daniel},
  title = {polynomial-neat: Polynomial NEAT in Rust},
  url = {https://github.com/dsgallups/polynomial-neat},
  year = {2024}
}
```

Original NEAT paper:
```bibtex
@article{stanley2002evolving,
  title={Evolving neural networks through augmenting topologies},
  author={Stanley, Kenneth O and Miikkulainen, Risto},
  journal={Evolutionary computation},
  volume={10},
  number={2},
  pages={99--127},
  year={2002},
  publisher={MIT Press}
}
```

## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
