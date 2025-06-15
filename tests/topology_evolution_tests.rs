//! Integration tests for network topology evolution.
//!
//! These tests verify that:
//! - Network topologies evolve correctly through mutations
//! - Evolution preserves network integrity and constraints
//! - Mutations produce valid, executable networks
//! - Edge cases in evolution are handled properly

use burn_neat::poly::prelude::*;
use burn_neat::poly::topology::mutation::MutationChances;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashSet;

/// Helper function to create a deterministic RNG
fn test_rng() -> StdRng {
    StdRng::seed_from_u64(54321)
}

#[test]
fn test_basic_topology_creation() {
    let mut rng = test_rng();
    let mutations = MutationChances::new(0);

    // Test various topology sizes
    let test_cases = vec![
        (1, 1),   // Minimal
        (5, 3),   // Medium
        (10, 10), // Large
    ];

    for (inputs, outputs) in test_cases {
        let topology = PolyNetworkTopology::new(inputs, outputs, mutations, &mut rng);

        // Verify neuron counts
        let neurons = topology.neurons();
        let input_count = neurons
            .iter()
            .filter(|n| n.read().unwrap().is_input())
            .count();
        let output_count = neurons
            .iter()
            .filter(|n| n.read().unwrap().is_output())
            .count();

        assert_eq!(input_count, inputs, "Should have {} input neurons", inputs);
        assert_eq!(
            output_count, outputs,
            "Should have {} output neurons",
            outputs
        );
    }
}

#[test]
fn test_topology_deep_clone() {
    let mut rng = test_rng();
    let mutations = MutationChances::new(50);
    let original = PolyNetworkTopology::new(3, 2, mutations, &mut rng);

    // Clone the topology
    let cloned = original.deep_clone();

    // Verify that neuron IDs are different
    let original_ids: HashSet<_> = original.neuron_ids().into_iter().collect();
    let cloned_ids: HashSet<_> = cloned.neuron_ids().into_iter().collect();

    // No ID should be shared between original and clone
    let intersection: HashSet<_> = original_ids.intersection(&cloned_ids).collect();
    assert!(
        intersection.is_empty(),
        "Cloned topology should have all new neuron IDs"
    );

    // But structure should be the same
    assert_eq!(
        original.neurons().len(),
        cloned.neurons().len(),
        "Cloned topology should have same number of neurons"
    );
}

#[test]
fn test_add_neuron_mutation() {
    let mut rng = test_rng();
    // Configure to only add neurons
    let mutations = MutationChances::new_from_raw(
        100,   // Always mutate
        100.0, // Only split connections (add neurons)
        0.0, 0.0, 0.0, 0.0,
    );

    let mut topology = PolyNetworkTopology::new_thoroughly_connected(2, 1, mutations, &mut rng);
    let initial_count = topology.neurons().len();

    // Evolve multiple times
    for _ in 0..5 {
        topology = topology.replicate(&mut rng);
    }

    let final_count = topology.neurons().len();
    assert!(
        final_count > initial_count,
        "Should have added neurons: {} -> {}",
        initial_count,
        final_count
    );

    // Verify the network is still valid
    let network = topology.to_simple_network();
    let outputs: Vec<f32> = network.predict(&[1.0, 1.0]).collect();
    assert_eq!(outputs.len(), 1);
    assert!(outputs[0].is_finite());
}

#[test]
fn test_add_connection_mutation() {
    let mut rng = test_rng();
    // Configure to only add connections
    let mutations = MutationChances::new_from_raw(
        100, // Always mutate
        0.0, 100.0, // Only add connections
        0.0, 0.0, 0.0,
    );

    // Start with a sparse network
    let mut topology = PolyNetworkTopology::new(4, 2, mutations, &mut rng);

    // Count initial connections
    let initial_connections = count_total_connections(&topology);

    // Evolve to add connections
    for _ in 0..10 {
        topology = topology.replicate(&mut rng);
    }

    let final_connections = count_total_connections(&topology);
    assert!(
        final_connections >= initial_connections,
        "Should have same or more connections: {} -> {}",
        initial_connections,
        final_connections
    );
}

#[test]
fn test_weight_mutation() {
    let mut rng = test_rng();
    // Configure to only mutate weights
    let mutations = MutationChances::new_from_raw(
        100, // Always mutate
        0.0, 0.0, 0.0, 100.0, // Only mutate weights
        0.0,
    );

    let topology = PolyNetworkTopology::new(2, 1, mutations, &mut rng);
    let network1 = topology.to_simple_network();

    // Get output before mutation
    let inputs = vec![1.5, 2.5];
    let output1: Vec<f32> = network1.predict(&inputs).collect();

    // Mutate weights
    let mutated = topology.replicate(&mut rng);
    let network2 = mutated.to_simple_network();
    let output2: Vec<f32> = network2.predict(&inputs).collect();

    // Outputs should differ due to weight changes
    // (May occasionally be the same if mutation didn't affect active connections)
    if (output1[0] - output2[0]).abs() < f32::EPSILON {
        // Try a few more times
        let mut found_difference = false;
        for _ in 0..5 {
            let mutated = topology.replicate(&mut rng);
            let network = mutated.to_simple_network();
            let output: Vec<f32> = network.predict(&inputs).collect();
            if (output1[0] - output[0]).abs() > f32::EPSILON {
                found_difference = true;
                break;
            }
        }
        assert!(
            found_difference,
            "Weight mutations should eventually change output"
        );
    }
}

#[test]
fn test_exponent_mutation() {
    let mut rng = test_rng();
    // Configure to only mutate exponents
    let mutations = MutationChances::new_from_raw(
        100, // Always mutate
        0.0, 0.0, 0.0, 0.0, 100.0, // Only mutate exponents
    );

    let topology = PolyNetworkTopology::new(2, 1, mutations, &mut rng);

    // Evolve multiple times to ensure some exponents change
    let mut evolved = topology.deep_clone();
    for _ in 0..10 {
        evolved = evolved.replicate(&mut rng);
    }

    // Both topologies should produce valid networks
    let network1 = topology.to_simple_network();
    let network2 = evolved.to_simple_network();

    let inputs = vec![2.0, 3.0];
    let output1: Vec<f32> = network1.predict(&inputs).collect();
    let output2: Vec<f32> = network2.predict(&inputs).collect();

    assert!(output1[0].is_finite());
    assert!(output2[0].is_finite());
}

#[test]
fn test_remove_neuron_mutation() {
    let mut rng = test_rng();
    // First, create a network with many hidden neurons
    let add_mutations = MutationChances::new_from_raw(
        100, 100.0, // Add neurons
        0.0, 0.0, 0.0, 0.0,
    );

    let mut topology = PolyNetworkTopology::new(3, 2, add_mutations, &mut rng);

    // Add some hidden neurons
    for _ in 0..10 {
        topology = topology.replicate(&mut rng);
    }

    let neurons_before_removal = topology.neurons().len();

    // Now configure to only remove neurons
    let remove_mutations = MutationChances::new_from_raw(
        100, 0.0, 0.0, 100.0, // Only remove neurons
        0.0, 0.0,
    );

    // Update mutation chances
    let neurons = topology.neurons().clone();
    let mutations = remove_mutations;
    let mut topology = PolyNetworkTopology::from_raw_parts(neurons, mutations);

    // Try to remove neurons
    for _ in 0..5 {
        topology = topology.replicate(&mut rng);
    }

    let neurons_after_removal = topology.neurons().len();

    // Should have same or fewer neurons (can't remove input/output neurons)
    assert!(
        neurons_after_removal <= neurons_before_removal,
        "Should have same or fewer neurons: {} -> {}",
        neurons_before_removal,
        neurons_after_removal
    );

    // Network should still be valid
    let network = topology.to_simple_network();
    assert_eq!(network.num_inputs(), 3);
    assert_eq!(network.num_outputs(), 2);
}

#[test]
fn test_evolution_preserves_io_neurons() {
    let mut rng = test_rng();
    let mutations = MutationChances::new(90); // High mutation rate

    let original = PolyNetworkTopology::new(5, 3, mutations, &mut rng);
    let mut topology = original.deep_clone();

    // Evolve aggressively
    for generation in 0..50 {
        topology = topology.replicate(&mut rng);

        // Count neuron types
        let neurons = topology.neurons();
        let input_count = neurons
            .iter()
            .filter(|n| n.read().unwrap().is_input())
            .count();
        let output_count = neurons
            .iter()
            .filter(|n| n.read().unwrap().is_output())
            .count();

        assert_eq!(
            input_count, 5,
            "Generation {} should maintain 5 inputs",
            generation
        );
        assert_eq!(
            output_count, 3,
            "Generation {} should maintain 3 outputs",
            generation
        );
    }
}

#[test]
fn test_thoroughly_connected_topology() {
    let mut rng = test_rng();
    let mutations = MutationChances::new(0);

    let topology = PolyNetworkTopology::new_thoroughly_connected(4, 3, mutations, &mut rng);

    // In a thoroughly connected network, each output should have connections from all inputs
    let neurons = topology.neurons();
    let output_neurons: Vec<_> = neurons
        .iter()
        .filter(|n| n.read().unwrap().is_output())
        .collect();

    assert_eq!(output_neurons.len(), 3);

    for output in output_neurons {
        let output_read = output.read().unwrap();
        if let Some(props) = output_read.props() {
            // Should have connections from all 4 inputs
            assert_eq!(
                props.num_inputs(),
                4,
                "Output neuron should connect to all 4 inputs"
            );
        } else {
            panic!("Output neuron should have properties");
        }
    }
}

#[test]
fn test_mutation_chances_evolution() {
    let mut rng = test_rng();
    // Create mutations that can evolve themselves
    let mut mutations = MutationChances::new(100); // High self-mutation

    let original = mutations;

    // Let mutation chances evolve
    for _ in 0..10 {
        mutations.adjust_mutation_chances(&mut rng);
    }

    // Chances should have changed
    let changed = mutations.split_connection() != original.split_connection()
        || mutations.add_connection() != original.add_connection()
        || mutations.remove_connection() != original.remove_connection()
        || mutations.mutate_weight() != original.mutate_weight()
        || mutations.mutate_exponent() != original.mutate_exponent();

    assert!(changed, "Mutation chances should evolve over time");

    // Should still sum to 100
    let total = mutations.split_connection()
        + mutations.add_connection()
        + mutations.remove_connection()
        + mutations.mutate_weight()
        + mutations.mutate_exponent();
    assert!((total - 100.0).abs() < 0.001);
}

#[test]
fn test_complex_evolution_scenario() {
    let mut rng = test_rng();
    // Use balanced mutations
    let mutations = MutationChances::new_from_raw(
        80,   // 80% chance to mutate
        30.0, // Add neurons
        25.0, // Add connections
        10.0, // Remove neurons
        25.0, // Mutate weights
        10.0, // Mutate exponents
    );

    let mut topology = PolyNetworkTopology::new(4, 2, mutations, &mut rng);

    // Track evolution statistics
    let mut stats = EvolutionStats::new();

    for _generation in 0..20 {
        let before = topology.neurons().len();
        topology = topology.replicate(&mut rng);
        let after = topology.neurons().len();

        stats.record_generation(before, after);

        // Verify network is still valid
        let network = topology.to_simple_network();
        let outputs: Vec<f32> = network.predict(&[1.0, 2.0, 3.0, 4.0]).collect();
        assert_eq!(outputs.len(), 2);
        assert!(outputs.iter().all(|&x| x.is_finite()));
    }

    // Should have seen some growth
    assert!(
        stats.max_neurons > stats.initial_neurons,
        "Network should have grown during evolution"
    );
}

#[test]
fn test_evolution_with_no_mutations() {
    let mut rng = test_rng();
    let mutations = MutationChances::none(); // No mutations

    let topology = PolyNetworkTopology::new(3, 2, mutations, &mut rng);
    let initial_neurons = topology.neurons().len();

    // Evolve multiple times
    let mut evolved = topology.deep_clone();
    for _ in 0..10 {
        evolved = evolved.replicate(&mut rng);
    }

    // Should be unchanged
    assert_eq!(
        evolved.neurons().len(),
        initial_neurons,
        "Topology should not change with zero mutation chances"
    );
}

#[test]
fn test_neuron_id_uniqueness() {
    let mut rng = test_rng();
    let mutations = MutationChances::new(80);

    let mut topology = PolyNetworkTopology::new(5, 3, mutations, &mut rng);

    // Evolve and collect all neuron IDs
    let mut all_ids = HashSet::new();

    for _ in 0..20 {
        topology = topology.replicate(&mut rng);

        for id in topology.neuron_ids() {
            // Each ID should be unique across all generations
            assert!(all_ids.insert(id), "Neuron ID {:?} was reused!", id);
        }
    }
}

#[test]
fn test_cyclic_connections() {
    let mut rng = test_rng();
    // High chance to add connections to create potential cycles
    let mutations = MutationChances::new_from_raw(
        100, 50.0, // Add neurons
        50.0, // Add connections
        0.0, 0.0, 0.0,
    );

    let mut topology = PolyNetworkTopology::new(2, 2, mutations, &mut rng);

    // Evolve to create complex topology
    for _ in 0..20 {
        topology = topology.replicate(&mut rng);
    }

    // Network should still work even with potential cycles
    let network = topology.to_simple_network();
    let outputs: Vec<f32> = network.predict(&[1.0, 1.0]).collect();
    assert_eq!(outputs.len(), 2);
    assert!(outputs.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_large_network_evolution() {
    let mut rng = test_rng();
    let mutations = MutationChances::new(70);

    // Start with a larger network
    let mut topology = PolyNetworkTopology::new(10, 5, mutations, &mut rng);

    // Evolve
    for _ in 0..10 {
        topology = topology.replicate(&mut rng);
    }

    // Should still produce valid network
    let network = topology.to_simple_network();
    assert_eq!(network.num_inputs(), 10);
    assert_eq!(network.num_outputs(), 5);

    let inputs = vec![1.0; 10];
    let outputs: Vec<f32> = network.predict(&inputs).collect();
    assert_eq!(outputs.len(), 5);
}

// Helper function to count total connections in a topology
fn count_total_connections(topology: &PolyNetworkTopology) -> usize {
    topology
        .neurons()
        .iter()
        .map(|neuron| {
            let n = neuron.read().unwrap();
            if let Some(inputs) = n.props().map(|p| p.inputs()) {
                inputs.len()
            } else {
                0
            }
        })
        .sum()
}

// Helper struct to track evolution statistics
struct EvolutionStats {
    initial_neurons: usize,
    max_neurons: usize,
    generations: Vec<(usize, usize)>,
}

impl EvolutionStats {
    fn new() -> Self {
        Self {
            initial_neurons: 0,
            max_neurons: 0,
            generations: Vec::new(),
        }
    }

    fn record_generation(&mut self, before: usize, after: usize) {
        if self.generations.is_empty() {
            self.initial_neurons = before;
            self.max_neurons = before;
        }

        self.generations.push((before, after));
        self.max_neurons = self.max_neurons.max(after);
    }
}
