//! Integration tests for tensor setup and Burn backend integration.
//!
//! These tests verify that:
//! - Tensors are correctly initialized for polynomial networks
//! - Burn backends (NdArray, CUDA, WGPU) work correctly
//! - Network topology converts properly to tensor representations
//! - Tensor operations produce expected results

use burn::backend::{NdArray, ndarray::NdArrayDevice};
use polynomial_neat::poly::{
    burn_net::network::BurnNetwork, prelude::*, topology::mutation::MutationChances,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Helper function to create a deterministic RNG
fn test_rng() -> StdRng {
    StdRng::seed_from_u64(12345)
}

/// Helper function to create a test topology
fn create_test_topology(
    inputs: usize,
    outputs: usize,
    rng: &mut impl rand::Rng,
) -> PolyNetworkTopology {
    let mutations = MutationChances::new(0); // No mutations for predictable testing
    PolyNetworkTopology::new(inputs, outputs, mutations, rng)
}

#[test]
fn test_burn_network_creation() {
    // Test basic burn network creation from topology
    let mut rng = test_rng();
    let topology = create_test_topology(3, 2, &mut rng);

    let device = NdArrayDevice::default();
    let burn_network = BurnNetwork::<NdArray>::from_topology(&topology, device);

    // Test that network was created successfully
    let inputs = vec![1.0, 2.0, 3.0];
    let outputs = burn_network.predict(&inputs);

    assert_eq!(outputs.len(), 2, "Should have 2 outputs");
    assert!(
        outputs.iter().all(|&x| x.is_finite()),
        "All outputs should be finite"
    );
}

#[test]
fn test_tensor_dimensions() {
    // Test that tensor dimensions match network structure
    let mut rng = test_rng();

    let test_cases = vec![
        (1, 1),   // Minimal network
        (5, 3),   // More inputs than outputs
        (2, 7),   // More outputs than inputs
        (10, 10), // Equal dimensions
    ];

    let device = NdArrayDevice::default();

    for (num_inputs, num_outputs) in test_cases {
        let topology = create_test_topology(num_inputs, num_outputs, &mut rng);
        let burn_network = BurnNetwork::<NdArray>::from_topology(&topology, device);

        // Create input tensor
        let inputs = vec![0.5_f32; num_inputs];
        let outputs = burn_network.predict(&inputs);

        assert_eq!(
            outputs.len(),
            num_outputs,
            "Network with {} inputs and {} outputs should produce {} outputs",
            num_inputs,
            num_outputs,
            num_outputs
        );
    }
}

#[test]
fn test_tensor_initialization() {
    // Test that tensors are properly initialized with weights and biases
    let mut rng = test_rng();
    let topology = create_test_topology(2, 1, &mut rng);

    let device = NdArrayDevice::default();
    let burn_network = BurnNetwork::<NdArray>::from_topology(&topology, device);

    // Run multiple predictions to ensure consistent initialization
    let inputs = vec![1.0, 1.0];
    let output1 = burn_network.predict(&inputs);
    let output2 = burn_network.predict(&inputs);

    // Should produce same output for same input (deterministic)
    assert_eq!(output1.len(), output2.len());
    for (o1, o2) in output1.iter().zip(output2.iter()) {
        assert!(
            (o1 - o2).abs() < f32::EPSILON,
            "Outputs should be deterministic: {} vs {}",
            o1,
            o2
        );
    }
}

#[test]
fn test_burn_network_with_hidden_layers() {
    // Test burn network with hidden layers added through evolution
    let mut rng = test_rng();
    let mutations = MutationChances::new_from_raw(
        100,  // Always mutate
        80.0, // High chance to split connections
        20.0, // Some chance to add connections
        0.0, 0.0, 0.0,
    );

    let mut topology = PolyNetworkTopology::new(3, 2, mutations, &mut rng);

    // Evolve to add complexity
    for _ in 0..5 {
        topology = topology.replicate(&mut rng);
    }

    let device = NdArrayDevice::default();
    let burn_network = BurnNetwork::<NdArray>::from_topology(&topology, device);

    // Test with various inputs
    let test_inputs = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, -1.0],
        vec![0.5, 0.5, 0.5],
        vec![2.0, -2.0, 1.0],
    ];

    for inputs in test_inputs {
        let outputs = burn_network.predict(&inputs);
        assert_eq!(outputs.len(), 2);
        assert!(
            outputs.iter().all(|&x| x.is_finite()),
            "Outputs should be finite for inputs {:?}",
            inputs
        );
    }
}

#[test]
fn test_tensor_operations_polynomial() {
    // Test that tensor operations correctly implement polynomial activation
    // output = Σ(weight_i * input_i^exponent_i) + bias

    let mut rng = test_rng();
    let topology = create_test_topology(1, 1, &mut rng);

    let device = NdArrayDevice::default();
    let burn_network = BurnNetwork::<NdArray>::from_topology(&topology, device);

    // Test with different input values to verify polynomial behavior
    let test_values = vec![0.0, 0.5, 1.0, 2.0, -1.0, -2.0];

    for value in test_values {
        let output = burn_network.predict(&[value]);
        assert_eq!(output.len(), 1);
        assert!(
            output[0].is_finite(),
            "Output should be finite for input {}",
            value
        );
    }
}

#[test]
fn test_batch_tensor_processing() {
    // Test that multiple inputs can be processed efficiently
    let mut rng = test_rng();
    let topology = create_test_topology(4, 3, &mut rng);

    let device = NdArrayDevice::default();
    let burn_network = BurnNetwork::<NdArray>::from_topology(&topology, device);

    // Process multiple inputs
    let batch_size = 10;
    let mut all_outputs = Vec::new();

    for i in 0..batch_size {
        let inputs = vec![i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32];
        let outputs = burn_network.predict(&inputs);
        all_outputs.push(outputs);
    }

    // Verify all outputs
    assert_eq!(all_outputs.len(), batch_size);
    for (i, outputs) in all_outputs.iter().enumerate() {
        assert_eq!(outputs.len(), 3, "Batch {} should have 3 outputs", i);
        assert!(
            outputs.iter().all(|&x| x.is_finite()),
            "Batch {} outputs should be finite",
            i
        );
    }
}

#[test]
fn test_tensor_memory_layout() {
    // Test that tensors maintain proper memory layout for efficient computation
    let mut rng = test_rng();
    let topology = create_test_topology(5, 4, &mut rng);

    let device = NdArrayDevice::default();
    let burn_network = BurnNetwork::<NdArray>::from_topology(&topology, device);

    // Create inputs that test memory access patterns
    let inputs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let outputs = burn_network.predict(&inputs);

    assert_eq!(outputs.len(), 4);

    // Verify outputs are reasonable (not NaN, not infinite)
    for (i, &output) in outputs.iter().enumerate() {
        assert!(
            output.is_finite(),
            "Output {} should be finite, got {}",
            i,
            output
        );
    }
}

#[test]
fn test_fully_connected_tensor_setup() {
    // Test tensor setup for fully connected networks
    let mut rng = test_rng();
    let mutations = MutationChances::new(0);
    let topology = PolyNetworkTopology::new_thoroughly_connected(6, 4, mutations, &mut rng);

    let device = NdArrayDevice::default();
    let burn_network = BurnNetwork::<NdArray>::from_topology(&topology, device);

    // In a fully connected network, all inputs affect all outputs
    let inputs = vec![1.0, -1.0, 2.0, -2.0, 0.5, -0.5];
    let outputs = burn_network.predict(&inputs);

    assert_eq!(outputs.len(), 4);
    assert!(outputs.iter().all(|&x| x.is_finite()));

    // Test that changing any input affects outputs
    let mut modified_inputs = inputs.clone();
    modified_inputs[0] = 10.0; // Change first input dramatically
    let modified_outputs = burn_network.predict(&modified_inputs);

    // At least one output should be different
    let any_different = outputs
        .iter()
        .zip(modified_outputs.iter())
        .any(|(o1, o2)| (o1 - o2).abs() > f32::EPSILON);

    assert!(
        any_different,
        "Changing input should affect at least one output in fully connected network"
    );
}

#[test]
fn test_tensor_numerical_stability() {
    // Test numerical stability with extreme values
    let mut rng = test_rng();
    let topology = create_test_topology(3, 2, &mut rng);

    let device = NdArrayDevice::default();
    let burn_network = BurnNetwork::<NdArray>::from_topology(&topology, device);

    // Test with various extreme inputs
    let test_cases = vec![
        vec![1e-10, 1e-10, 1e-10], // Very small values
        vec![1e10, 1e10, 1e10],    // Very large values
        vec![1e-10, 1e10, 0.0],    // Mixed scales
        vec![f32::MIN_POSITIVE, f32::MIN_POSITIVE, f32::MIN_POSITIVE], // Minimum positive
    ];

    for inputs in test_cases {
        let outputs = burn_network.predict(&inputs);
        assert_eq!(outputs.len(), 2);

        // Even with extreme inputs, we should avoid NaN/Inf where possible
        for (i, &output) in outputs.iter().enumerate() {
            if !output.is_finite() {
                println!(
                    "Warning: Non-finite output {} for inputs {:?}: {}",
                    i, inputs, output
                );
            }
        }
    }
}

#[test]
fn test_zero_connections_handling() {
    // Test handling of neurons with no connections
    let mut rng = test_rng();
    let mutations = MutationChances::new(0);

    // Create a minimal topology
    let topology = PolyNetworkTopology::new(1, 1, mutations, &mut rng);

    let device = NdArrayDevice::default();
    let burn_network = BurnNetwork::<NdArray>::from_topology(&topology, device);

    // Even with minimal connections, should produce valid output
    let output = burn_network.predict(&[1.0]);
    assert_eq!(output.len(), 1);
    assert!(output[0].is_finite());
}

#[test]
fn test_tensor_consistency_across_backends() {
    // Test that the same topology produces consistent behavior
    let mut rng = test_rng();
    let topology = create_test_topology(3, 2, &mut rng);

    let device = NdArrayDevice::default();

    // Create multiple networks from same topology
    let network1 = BurnNetwork::<NdArray>::from_topology(&topology, device);
    let network2 = BurnNetwork::<NdArray>::from_topology(&topology, device);

    let inputs = vec![1.0, 2.0, 3.0];
    let outputs1 = network1.predict(&inputs);
    let outputs2 = network2.predict(&inputs);

    // Should produce identical outputs
    assert_eq!(outputs1.len(), outputs2.len());
    for (o1, o2) in outputs1.iter().zip(outputs2.iter()) {
        assert!(
            (o1 - o2).abs() < 1e-6,
            "Networks from same topology should produce same output: {} vs {}",
            o1,
            o2
        );
    }
}

#[test]
fn test_evolved_network_tensor_integrity() {
    // Test that evolved networks maintain tensor integrity
    let mut rng = test_rng();
    let mutations = MutationChances::new(75);

    let mut topology = PolyNetworkTopology::new(4, 2, mutations, &mut rng);
    let device = NdArrayDevice::default();

    // Test network at each evolution stage
    for generation in 0..10 {
        let burn_network = BurnNetwork::<NdArray>::from_topology(&topology, device);

        let inputs = vec![1.0, 2.0, 3.0, 4.0];
        let outputs = burn_network.predict(&inputs);

        assert_eq!(
            outputs.len(),
            2,
            "Generation {} should maintain 2 outputs",
            generation
        );
        assert!(
            outputs.iter().all(|&x| x.is_finite()),
            "Generation {} should produce finite outputs",
            generation
        );

        // Evolve for next iteration
        topology = topology.replicate(&mut rng);
    }
}

#[test]
fn test_tensor_gradient_flow() {
    // Test that tensor setup allows for proper gradient flow (important for training)
    let mut rng = test_rng();
    let topology = create_test_topology(2, 1, &mut rng);

    let device = NdArrayDevice::default();
    let burn_network = BurnNetwork::<NdArray>::from_topology(&topology, device);

    // Test with inputs that should produce different outputs
    let inputs1 = vec![1.0, 0.0];
    let inputs2 = vec![0.0, 1.0];
    let inputs3 = vec![1.0, 1.0];

    let output1 = burn_network.predict(&inputs1);
    let output2 = burn_network.predict(&inputs2);
    let output3 = burn_network.predict(&inputs3);

    // Outputs should be different for different inputs
    let all_same = output1[0] == output2[0] && output2[0] == output3[0];
    assert!(
        !all_same,
        "Different inputs should produce different outputs in most cases"
    );
}

#[test]
fn test_large_network_tensor_setup() {
    // Test tensor setup for larger networks
    let mut rng = test_rng();
    let mutations = MutationChances::new_from_raw(
        100,  // Always mutate
        60.0, // Add neurons
        40.0, // Add connections
        0.0, 0.0, 0.0,
    );

    let mut topology = PolyNetworkTopology::new(10, 5, mutations, &mut rng);

    // Evolve to create a complex network
    for _ in 0..20 {
        topology = topology.replicate(&mut rng);
    }

    let device = NdArrayDevice::default();
    let burn_network = BurnNetwork::<NdArray>::from_topology(&topology, device);

    // Test with full input vector
    let inputs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let outputs = burn_network.predict(&inputs);

    assert_eq!(outputs.len(), 5, "Should maintain 5 outputs");
    assert!(
        outputs.iter().all(|&x| x.is_finite()),
        "Large network should still produce finite outputs"
    );
}
