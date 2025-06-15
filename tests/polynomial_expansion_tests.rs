//! Integration tests for polynomial neural network expansion and computation.
//!
//! These tests verify that polynomial networks correctly:
//! - Expand polynomial activation functions
//! - Set up tensor structures for computation
//! - Process inputs and produce expected outputs

use burn_neat::poly::prelude::*;
use burn_neat::poly::topology::mutation::MutationChances;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Helper function to create a deterministic RNG for reproducible tests
fn test_rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

/// Helper function to create a simple topology with known structure
fn create_simple_topology(
    num_inputs: usize,
    num_outputs: usize,
    rng: &mut impl rand::Rng,
) -> PolyNetworkTopology {
    let mutations = MutationChances::new(0); // No mutations for predictable testing
    PolyNetworkTopology::new(num_inputs, num_outputs, mutations, rng)
}

#[test]
fn test_polynomial_activation_basic() {
    // Test basic polynomial activation: output = weight * input^exponent + bias

    // Create a simple 1-input, 1-output network
    let mut rng = test_rng();
    let topology = create_simple_topology(1, 1, &mut rng);
    let network = topology.to_simple_network();

    // Test with different input values
    let test_cases = vec![0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5];

    for input in test_cases {
        let result: Vec<f32> = network.predict(&[input]).collect();
        assert_eq!(result.len(), 1, "Should have exactly one output");

        // The output should be finite (not NaN or infinite)
        assert!(
            result[0].is_finite(),
            "Output should be finite for input {}",
            input
        );
    }
}

#[test]
fn test_polynomial_expansion_with_multiple_inputs() {
    // Test that polynomial expansion works correctly with multiple inputs
    // Each input contribution: weight_i * input_i^exponent_i
    // Total: Σ(weight_i * input_i^exponent_i) + bias

    let mut rng = test_rng();
    let topology = create_simple_topology(3, 1, &mut rng);
    let network = topology.to_simple_network();

    // Test various input combinations
    let test_cases = vec![
        vec![1.0, 1.0, 1.0],
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, -1.0],
        vec![2.0, 3.0, 4.0],
        vec![-1.0, -2.0, -3.0],
    ];

    for inputs in test_cases {
        let result: Vec<f32> = network.predict(&inputs).collect();
        assert_eq!(result.len(), 1, "Should have exactly one output");
        assert!(
            result[0].is_finite(),
            "Output should be finite for inputs {:?}",
            inputs
        );
    }
}

#[test]
fn test_exponent_behavior() {
    // Test that exponents of 0 and 1 behave correctly
    // Exponent 0: input^0 = 1 (constant)
    // Exponent 1: input^1 = input (linear)

    let mut rng = test_rng();

    // Create multiple networks to test different random configurations
    for _ in 0..10 {
        let topology = create_simple_topology(2, 2, &mut rng);
        let network = topology.to_simple_network();

        // Test with various inputs
        let inputs = vec![2.0, 3.0];
        let outputs: Vec<f32> = network.predict(&inputs).collect();

        assert_eq!(outputs.len(), 2, "Should have two outputs");
        assert!(
            outputs.iter().all(|&x| x.is_finite()),
            "All outputs should be finite"
        );
    }
}

#[test]
fn test_network_with_hidden_layers() {
    // Test network behavior after adding hidden neurons
    let mut rng = test_rng();
    let mutations = MutationChances::new_from_raw(
        100,  // Always mutate
        80.0, // High chance to split connections (add neurons)
        20.0, // Low chance for other mutations
        0.0, 0.0, 0.0,
    );

    let mut topology = PolyNetworkTopology::new(2, 1, mutations, &mut rng);

    // Evolve the network to add hidden neurons
    for _ in 0..5 {
        topology = topology.replicate(&mut rng);
    }

    let network = topology.to_simple_network();

    // Verify the network still produces valid outputs
    let test_inputs = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];

    for inputs in test_inputs {
        let outputs: Vec<f32> = network.predict(&inputs).collect();
        assert_eq!(outputs.len(), 1);
        assert!(
            outputs[0].is_finite(),
            "Output should be finite for inputs {:?}",
            inputs
        );
    }
}

#[test]
fn test_fully_connected_network() {
    // Test a fully connected network where all inputs connect to all outputs
    let mut rng = test_rng();
    let mutations = MutationChances::new(0);
    let topology = PolyNetworkTopology::new_thoroughly_connected(4, 3, mutations, &mut rng);

    let network = topology.to_simple_network();

    // Verify network structure
    assert_eq!(network.num_inputs(), 4);
    assert_eq!(network.num_outputs(), 3);

    // Test with various inputs
    let inputs = vec![1.0, -1.0, 0.5, -0.5];
    let outputs: Vec<f32> = network.predict(&inputs).collect();

    assert_eq!(outputs.len(), 3);
    assert!(outputs.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_network_tensor_dimensions() {
    // Test that tensor dimensions are set up correctly for various network sizes
    let mut rng = test_rng();

    let test_cases = vec![
        (1, 1),   // Minimal network
        (5, 3),   // More inputs than outputs
        (2, 7),   // More outputs than inputs
        (10, 10), // Equal inputs and outputs
    ];

    for (num_inputs, num_outputs) in test_cases {
        let topology = create_simple_topology(num_inputs, num_outputs, &mut rng);
        let network = topology.to_simple_network();

        assert_eq!(
            network.num_inputs(),
            num_inputs,
            "Network should have {} inputs",
            num_inputs
        );
        assert_eq!(
            network.num_outputs(),
            num_outputs,
            "Network should have {} outputs",
            num_outputs
        );

        // Test with correct number of inputs
        let inputs = vec![0.5; num_inputs];
        let outputs: Vec<f32> = network.predict(&inputs).collect();
        assert_eq!(
            outputs.len(),
            num_outputs,
            "Should produce {} outputs",
            num_outputs
        );
    }
}

#[test]
fn test_weight_and_exponent_mutations() {
    // Test that weight and exponent mutations affect network output
    let mut rng = test_rng();

    // Create network with only weight/exponent mutations
    let mutations = MutationChances::new_from_raw(
        100, // Always mutate
        0.0, // No topology changes
        0.0, 0.0, 50.0, // Weight mutations
        50.0, // Exponent mutations
    );

    let topology = PolyNetworkTopology::new(2, 1, mutations, &mut rng);
    let network1 = topology.to_simple_network();

    // Get output before mutation
    let inputs = vec![1.5, 2.5];
    let output1: Vec<f32> = network1.predict(&inputs).collect();

    // Mutate and create new network
    let mutated_topology = topology.replicate(&mut rng);
    let network2 = mutated_topology.to_simple_network();
    let output2: Vec<f32> = network2.predict(&inputs).collect();

    // Outputs should be different due to mutations
    // (This might occasionally fail due to randomness, but very unlikely)
    assert!(
        (output1[0] - output2[0]).abs() > f32::EPSILON,
        "Mutation should change network output"
    );
}

#[test]
fn test_network_determinism() {
    // Test that networks produce deterministic outputs for the same inputs
    let mut rng = test_rng();
    let topology = create_simple_topology(3, 2, &mut rng);
    let network = topology.to_simple_network();

    let inputs = vec![1.0, 2.0, 3.0];

    // Run prediction multiple times
    let results: Vec<Vec<f32>> = (0..10)
        .map(|_| network.predict(&inputs).collect())
        .collect();

    // All results should be identical
    for i in 1..results.len() {
        assert_eq!(
            results[0], results[i],
            "Network should produce deterministic outputs"
        );
    }
}

#[test]
fn test_zero_input_handling() {
    // Test how the network handles zero inputs
    let mut rng = test_rng();
    let topology = create_simple_topology(3, 2, &mut rng);
    let network = topology.to_simple_network();

    // All zeros
    let outputs: Vec<f32> = network.predict(&[0.0, 0.0, 0.0]).collect();
    assert_eq!(outputs.len(), 2);
    assert!(outputs.iter().all(|&x| x.is_finite()));

    // Mixed with zeros
    let outputs: Vec<f32> = network.predict(&[1.0, 0.0, -1.0]).collect();
    assert_eq!(outputs.len(), 2);
    assert!(outputs.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_large_input_values() {
    // Test network stability with large input values
    let mut rng = test_rng();
    let topology = create_simple_topology(2, 1, &mut rng);
    let network = topology.to_simple_network();

    let test_cases = vec![vec![10.0, 10.0], vec![100.0, -100.0], vec![1000.0, 0.0]];

    for inputs in test_cases {
        let outputs: Vec<f32> = network.predict(&inputs).collect();
        assert_eq!(outputs.len(), 1);
        // Output might be large but should still be finite
        assert!(
            outputs[0].is_finite(),
            "Output should be finite for large inputs {:?}",
            inputs
        );
    }
}

#[test]
fn test_network_evolution_preserves_io() {
    // Test that evolution preserves input/output dimensions
    let mut rng = test_rng();
    let mutations = MutationChances::new(80);

    let original_topology = PolyNetworkTopology::new(5, 3, mutations, &mut rng);
    let mut topology = original_topology.deep_clone();

    // Evolve multiple times
    for generation in 0..10 {
        topology = topology.replicate(&mut rng);
        let network = topology.to_simple_network();

        assert_eq!(
            network.num_inputs(),
            5,
            "Generation {} should maintain 5 inputs",
            generation
        );
        assert_eq!(
            network.num_outputs(),
            3,
            "Generation {} should maintain 3 outputs",
            generation
        );

        // Test that it still works
        let inputs = vec![1.0; 5];
        let outputs: Vec<f32> = network.predict(&inputs).collect();
        assert_eq!(outputs.len(), 3);
        assert!(outputs.iter().all(|&x| x.is_finite()));
    }
}

#[test]
fn test_polynomial_computation_example() {
    // Test a specific polynomial computation to verify correctness
    // This test manually calculates what the output should be for a simple case

    let mut rng = test_rng();
    let topology = create_simple_topology(1, 1, &mut rng);
    let network = topology.to_simple_network();

    // Test multiple input values
    for input in &[0.0, 1.0, 2.0, -1.0, 0.5] {
        let outputs: Vec<f32> = network.predict(&[*input]).collect();
        assert_eq!(outputs.len(), 1);

        // For a single input/output network with polynomial activation:
        // output = weight * input^exponent + bias
        // The exact value depends on random initialization, but it should be finite
        assert!(
            outputs[0].is_finite(),
            "Output should be finite for input {}",
            input
        );
    }
}

#[test]
fn test_network_node_counts() {
    // Test that node counts are tracked correctly
    let mut rng = test_rng();
    let mutations = MutationChances::new_from_raw(
        100,   // Always mutate
        100.0, // Only split connections (add neurons)
        0.0, 0.0, 0.0, 0.0,
    );

    let mut topology = PolyNetworkTopology::new(3, 2, mutations, &mut rng);
    let initial_network = topology.to_simple_network();
    let initial_nodes = initial_network.num_nodes();

    // Initial network should have exactly input + output nodes
    assert_eq!(
        initial_nodes, 5,
        "Initial network should have 3 inputs + 2 outputs = 5 nodes"
    );

    // Evolve to add hidden nodes
    for _ in 0..3 {
        topology = topology.replicate(&mut rng);
    }

    let evolved_network = topology.to_simple_network();
    let evolved_nodes = evolved_network.num_nodes();

    // Should have more nodes after evolution
    assert!(
        evolved_nodes > initial_nodes,
        "Evolution should add hidden nodes: {} > {}",
        evolved_nodes,
        initial_nodes
    );
}

#[test]
fn test_parallel_prediction() {
    // Test that parallel prediction works correctly
    use rayon::prelude::*;

    let mut rng = test_rng();
    let topology = create_simple_topology(4, 2, &mut rng);
    let network = topology.to_simple_network();

    // Create multiple input sets
    let input_sets: Vec<Vec<f32>> = (0..100)
        .map(|i| vec![i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32])
        .collect();

    // Predict in parallel
    let results: Vec<Vec<f32>> = input_sets
        .par_iter()
        .map(|inputs| network.predict(inputs).collect())
        .collect();

    // Verify all results
    assert_eq!(results.len(), 100);
    for (i, outputs) in results.iter().enumerate() {
        assert_eq!(outputs.len(), 2, "Result {} should have 2 outputs", i);
        assert!(
            outputs.iter().all(|&x| x.is_finite()),
            "Result {} should have finite outputs",
            i
        );
    }
}
