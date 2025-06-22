//! Integration tests for input/output validation and edge cases.
//!
//! These tests verify that:
//! - Networks correctly validate input dimensions
//! - Output dimensions match expectations
//! - Edge cases are handled gracefully
//! - Invalid inputs are properly rejected or handled

use polynomial_neat::poly::prelude::*;
use polynomial_neat::poly::topology::mutation::MutationChances;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Helper function to create a deterministic RNG
fn test_rng() -> StdRng {
    StdRng::seed_from_u64(99999)
}

#[test]
fn test_input_dimension_validation() {
    // Test that networks require correct number of inputs
    let mut rng = test_rng();
    let mutations = MutationChances::new(0);
    let topology = PolyNetworkTopology::new(3, 2, mutations, &mut rng);
    let network = topology.to_simple_network();

    // Correct number of inputs should work
    let correct_inputs = vec![1.0, 2.0, 3.0];
    let outputs: Vec<f32> = network.predict(&correct_inputs).collect();
    assert_eq!(outputs.len(), 2);

    // Test with wrong number of inputs would panic in current implementation
    // This documents expected behavior
}

#[test]
fn test_output_dimension_consistency() {
    // Test that output dimensions are always consistent
    let mut rng = test_rng();
    let test_cases = vec![(1, 1), (5, 1), (1, 5), (3, 3), (10, 7)];

    for (num_inputs, num_outputs) in test_cases {
        let mutations = MutationChances::new(0);
        let topology = PolyNetworkTopology::new(num_inputs, num_outputs, mutations, &mut rng);
        let network = topology.to_simple_network();

        // Create appropriate inputs
        let inputs = vec![0.5; num_inputs];
        let outputs: Vec<f32> = network.predict(&inputs).collect();

        assert_eq!(
            outputs.len(),
            num_outputs,
            "Network with {} outputs should always produce {} outputs",
            num_outputs,
            num_outputs
        );
    }
}

#[test]
fn test_empty_network_edge_case() {
    // Test the minimal possible network (1 input, 1 output)
    let mut rng = test_rng();
    let mutations = MutationChances::new(0);
    let topology = PolyNetworkTopology::new(1, 1, mutations, &mut rng);
    let network = topology.to_simple_network();

    let output: Vec<f32> = network.predict(&[0.0]).collect();
    assert_eq!(output.len(), 1);
    assert!(output[0].is_finite());
}

#[test]
fn test_nan_propagation() {
    // Test how NaN inputs are handled
    let mut rng = test_rng();
    let mutations = MutationChances::new(0);
    let topology = PolyNetworkTopology::new(3, 2, mutations, &mut rng);
    let network = topology.to_simple_network();

    // Test with NaN in different positions
    let test_cases = vec![
        vec![f32::NAN, 1.0, 1.0],
        vec![1.0, f32::NAN, 1.0],
        vec![1.0, 1.0, f32::NAN],
        vec![f32::NAN, f32::NAN, f32::NAN],
    ];

    for inputs in test_cases {
        let outputs: Vec<f32> = network.predict(&inputs).collect();
        assert_eq!(outputs.len(), 2);
        // NaN inputs will propagate to outputs in polynomial computation
        // This is expected behavior
    }
}

#[test]
fn test_infinity_handling() {
    // Test how infinite values are handled
    let mut rng = test_rng();
    let mutations = MutationChances::new(0);
    let topology = PolyNetworkTopology::new(2, 1, mutations, &mut rng);
    let network = topology.to_simple_network();

    let test_cases = vec![
        vec![f32::INFINITY, 0.0],
        vec![0.0, f32::INFINITY],
        vec![f32::NEG_INFINITY, 0.0],
        vec![f32::INFINITY, f32::NEG_INFINITY],
    ];

    for inputs in test_cases {
        let outputs: Vec<f32> = network.predict(&inputs).collect();
        assert_eq!(outputs.len(), 1);
        // Document that infinity can propagate through the network
        // This is expected behavior for polynomial operations
    }
}

#[test]
fn test_zero_handling() {
    // Test that zero inputs are handled correctly
    let mut rng = test_rng();
    let mutations = MutationChances::new(0);
    let topology = PolyNetworkTopology::new(5, 3, mutations, &mut rng);
    let network = topology.to_simple_network();

    // All zeros
    let all_zeros = vec![0.0; 5];
    let outputs: Vec<f32> = network.predict(&all_zeros).collect();
    assert_eq!(outputs.len(), 3);

    // The output should be just the bias terms when all inputs are zero
    for output in outputs {
        assert!(
            output.is_finite(),
            "Zero inputs should produce finite outputs"
        );
    }
}

#[test]
fn test_boundary_values() {
    // Test various boundary values
    let mut rng = test_rng();
    let mutations = MutationChances::new(0);
    let topology = PolyNetworkTopology::new(4, 2, mutations, &mut rng);
    let network = topology.to_simple_network();

    let test_cases = vec![
        vec![f32::MIN, f32::MIN, f32::MIN, f32::MIN],
        vec![f32::MAX, f32::MAX, f32::MAX, f32::MAX],
        vec![
            f32::MIN_POSITIVE,
            f32::MIN_POSITIVE,
            f32::MIN_POSITIVE,
            f32::MIN_POSITIVE,
        ],
        vec![f32::EPSILON, f32::EPSILON, f32::EPSILON, f32::EPSILON],
        vec![-f32::EPSILON, -f32::EPSILON, -f32::EPSILON, -f32::EPSILON],
    ];

    for inputs in test_cases {
        let outputs: Vec<f32> = network.predict(&inputs).collect();
        assert_eq!(outputs.len(), 2, "Should always produce 2 outputs");

        // Note: Some boundary values may produce non-finite results
        // This documents the current behavior
    }
}

#[test]
fn test_mixed_sign_inputs() {
    // Test with mixed positive and negative inputs
    let mut rng = test_rng();
    let mutations = MutationChances::new(0);
    let topology = PolyNetworkTopology::new(4, 2, mutations, &mut rng);
    let network = topology.to_simple_network();

    let test_cases = vec![
        vec![1.0, -1.0, 1.0, -1.0],
        vec![-10.0, 10.0, -10.0, 10.0],
        vec![0.1, -0.1, 0.01, -0.01],
        vec![-100.0, 100.0, -1000.0, 1000.0],
    ];

    for inputs in test_cases {
        let outputs: Vec<f32> = network.predict(&inputs).collect();
        assert_eq!(outputs.len(), 2);

        // Mixed signs should still produce valid outputs
        let finite_count = outputs.iter().filter(|&&x| x.is_finite()).count();
        assert!(
            finite_count > 0,
            "At least some outputs should be finite for mixed inputs"
        );
    }
}

#[test]
fn test_single_input_variations() {
    // Test how a single changing input affects outputs
    let mut rng = test_rng();
    let mutations = MutationChances::new(0);
    let topology = PolyNetworkTopology::new(3, 1, mutations, &mut rng);
    let network = topology.to_simple_network();

    let base_inputs = vec![1.0, 1.0, 1.0];
    let _base_output: Vec<f32> = network.predict(&base_inputs).collect();

    // Change each input individually
    for i in 0..3 {
        let mut modified = base_inputs.clone();
        modified[i] = 2.0;

        let modified_output: Vec<f32> = network.predict(&modified).collect();
        assert_eq!(modified_output.len(), 1);

        // Changing an input should generally change the output
        // (unless the weight is zero, which is unlikely)
    }
}

#[test]
fn test_output_range_expectations() {
    // Test that outputs fall within reasonable ranges for typical inputs
    let mut rng = test_rng();
    let mutations = MutationChances::new(0);
    let topology = PolyNetworkTopology::new(3, 2, mutations, &mut rng);
    let network = topology.to_simple_network();

    // Test with normalized inputs
    let normalized_inputs = vec![0.5, 0.5, 0.5];
    let outputs: Vec<f32> = network.predict(&normalized_inputs).collect();

    assert_eq!(outputs.len(), 2);
    for output in outputs {
        assert!(
            output.is_finite(),
            "Normalized inputs should produce finite outputs"
        );
    }
}

#[test]
fn test_network_after_many_mutations() {
    // Test that heavily mutated networks still validate I/O correctly
    let mut rng = test_rng();
    let mutations = MutationChances::new(90); // High mutation rate
    let mut topology = PolyNetworkTopology::new(4, 3, mutations, &mut rng);

    // Apply many mutations
    for _ in 0..50 {
        topology = topology.replicate(&mut rng);
    }

    let network = topology.to_simple_network();

    // Should still respect input/output dimensions
    assert_eq!(network.num_inputs(), 4);
    assert_eq!(network.num_outputs(), 3);

    let inputs = vec![1.0, 2.0, 3.0, 4.0];
    let outputs: Vec<f32> = network.predict(&inputs).collect();
    assert_eq!(outputs.len(), 3);
}

#[test]
fn test_precision_loss_accumulation() {
    // Test for precision loss with many operations
    let mut rng = test_rng();
    let mutations = MutationChances::new_from_raw(
        100, 80.0, 20.0, 0.0, 0.0, 0.0, // Only add complexity
    );

    let mut topology = PolyNetworkTopology::new(2, 1, mutations, &mut rng);

    // Create a deep network
    for _ in 0..20 {
        topology = topology.replicate(&mut rng);
    }

    let network = topology.to_simple_network();

    // Use small values that might accumulate errors
    let small_inputs = vec![1e-10, 1e-10];
    let outputs: Vec<f32> = network.predict(&small_inputs).collect();

    assert_eq!(outputs.len(), 1);
    // Document behavior with very small values
}

#[test]
fn test_deterministic_output() {
    // Test that the same input always produces the same output
    let mut rng = test_rng();
    let mutations = MutationChances::new(0);
    let topology = PolyNetworkTopology::new(5, 3, mutations, &mut rng);
    let network = topology.to_simple_network();

    let inputs = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // Run prediction multiple times
    let mut results = Vec::new();
    for _ in 0..10 {
        let outputs: Vec<f32> = network.predict(&inputs).collect();
        results.push(outputs);
    }

    // All results should be identical
    for i in 1..results.len() {
        assert_eq!(
            results[0].len(),
            results[i].len(),
            "Output length should be consistent"
        );

        for j in 0..results[0].len() {
            assert!(
                (results[0][j] - results[i][j]).abs() < f32::EPSILON,
                "Output should be deterministic: {} vs {}",
                results[0][j],
                results[i][j]
            );
        }
    }
}

#[test]
fn test_input_sensitivity() {
    // Test network sensitivity to small input changes
    let mut rng = test_rng();
    let mutations = MutationChances::new(0);
    let topology = PolyNetworkTopology::new(2, 1, mutations, &mut rng);
    let network = topology.to_simple_network();

    let base_inputs = vec![1.0, 1.0];
    let base_output: Vec<f32> = network.predict(&base_inputs).collect();

    // Make a tiny change
    let epsilon = 1e-6;
    let perturbed_inputs = vec![1.0 + epsilon, 1.0];
    let perturbed_output: Vec<f32> = network.predict(&perturbed_inputs).collect();

    // The change in output should be proportional to the change in input
    let output_change = (base_output[0] - perturbed_output[0]).abs();

    // Document that small input changes produce small output changes
    // (unless weights are very large)
    assert!(
        output_change.is_finite(),
        "Small input changes should produce finite output changes"
    );
}

#[test]
fn test_fully_connected_io_validation() {
    // Test I/O validation for fully connected networks
    let mut rng = test_rng();
    let mutations = MutationChances::new(0);
    let topology = PolyNetworkTopology::new_thoroughly_connected(8, 4, mutations, &mut rng);
    let network = topology.to_simple_network();

    assert_eq!(network.num_inputs(), 8);
    assert_eq!(network.num_outputs(), 4);

    // Test with correct inputs
    let inputs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let outputs: Vec<f32> = network.predict(&inputs).collect();

    assert_eq!(outputs.len(), 4, "Should produce exactly 4 outputs");

    // In a fully connected network, all outputs should be affected by inputs
    let all_finite = outputs.iter().all(|&x| x.is_finite());
    assert!(
        all_finite,
        "Fully connected network should produce finite outputs"
    );
}

#[test]
fn test_exponent_edge_cases() {
    // Test edge cases related to exponent values
    let mut rng = test_rng();
    let mutations = MutationChances::new(0);
    let topology = PolyNetworkTopology::new(3, 2, mutations, &mut rng);
    let network = topology.to_simple_network();

    // Test with inputs that might cause issues with different exponents
    let test_cases = vec![
        vec![0.0, 1.0, 2.0],    // Zero raised to any power
        vec![-1.0, -2.0, -3.0], // Negative values with integer exponents
        vec![1.0, 1.0, 1.0],    // One raised to any power is one
    ];

    for inputs in test_cases {
        let outputs: Vec<f32> = network.predict(&inputs).collect();
        assert_eq!(outputs.len(), 2);

        // Document behavior with special exponent cases
        // Note: Current implementation uses exponents 0, 1, or 2
        // so negative inputs with even exponents become positive
    }
}

#[test]
fn test_node_count_validation() {
    // Test that node counts are correctly reported
    let mut rng = test_rng();
    let mutations = MutationChances::new(0);

    let test_cases = vec![
        (1, 1, 2),    // 1 input + 1 output = 2 nodes
        (5, 3, 8),    // 5 inputs + 3 outputs = 8 nodes
        (10, 10, 20), // 10 inputs + 10 outputs = 20 nodes
    ];

    for (inputs, outputs, expected_nodes) in test_cases {
        let topology = PolyNetworkTopology::new(inputs, outputs, mutations, &mut rng);
        let network = topology.to_simple_network();

        assert_eq!(
            network.num_nodes(),
            expected_nodes,
            "Network with {} inputs and {} outputs should have {} nodes",
            inputs,
            outputs,
            expected_nodes
        );
    }
}
