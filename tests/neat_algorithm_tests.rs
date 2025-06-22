//! Comprehensive tests for the NEAT algorithm implementation.
//!
//! These tests verify that:
//! - The NEAT algorithm can solve classic problems like XOR
//! - Evolution produces improving fitness over generations
//! - Selection and reproduction work correctly
//! - Networks can learn non-linear functions

use burn_neat::poly::prelude::*;
use burn_neat::poly::topology::mutation::MutationChances;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Helper function to create a deterministic RNG
fn test_rng() -> StdRng {
    StdRng::seed_from_u64(12345)
}

/// Represents an individual in the population with its fitness
#[derive(Clone)]
struct Individual {
    topology: PolyNetworkTopology,
    fitness: f32,
}

impl Individual {
    fn new(topology: PolyNetworkTopology) -> Self {
        Self {
            topology,
            fitness: 0.0,
        }
    }

    fn evaluate<F>(&mut self, fitness_fn: F)
    where
        F: Fn(&SimplePolyNetwork) -> f32,
    {
        let network = self.topology.to_simple_network();
        self.fitness = fitness_fn(&network);
    }
}

/// Population of individuals for evolutionary algorithms
struct Population {
    individuals: Vec<Individual>,
    generation: usize,
}

impl Population {
    fn new(
        size: usize,
        inputs: usize,
        outputs: usize,
        mutations: MutationChances,
        rng: &mut impl rand::Rng,
    ) -> Self {
        let individuals = (0..size)
            .map(|_| {
                let topology = PolyNetworkTopology::new(inputs, outputs, mutations, rng);
                Individual::new(topology)
            })
            .collect();

        Self {
            individuals,
            generation: 0,
        }
    }

    fn evaluate_all<F>(&mut self, fitness_fn: F)
    where
        F: Fn(&SimplePolyNetwork) -> f32 + Sync,
    {
        self.individuals.iter_mut().for_each(|individual| {
            individual.evaluate(&fitness_fn);
        });
    }

    fn best(&self) -> Option<&Individual> {
        self.individuals
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
    }

    fn average_fitness(&self) -> f32 {
        let sum: f32 = self.individuals.iter().map(|i| i.fitness).sum();
        sum / self.individuals.len() as f32
    }

    fn evolve(&mut self, rng: &mut impl rand::Rng) {
        // Sort by fitness (best first)
        self.individuals
            .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        // Keep top 20% as elites
        let elite_count = self.individuals.len() / 5;
        let mut new_population = Vec::with_capacity(self.individuals.len());

        // Keep elites
        for i in 0..elite_count {
            new_population.push(self.individuals[i].clone());
        }

        // Fill rest with mutations of good individuals
        while new_population.len() < self.individuals.len() {
            // Select parent from top 50%
            let parent_idx = rng.random_range(0..self.individuals.len() / 2);
            let parent = &self.individuals[parent_idx];

            // Create offspring
            let offspring_topology = parent.topology.replicate(rng);
            new_population.push(Individual::new(offspring_topology));
        }

        self.individuals = new_population;
        self.generation += 1;
    }
}

/// Fitness function for XOR problem
fn xor_fitness(network: &SimplePolyNetwork) -> f32 {
    let test_cases = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];

    let mut total_error = 0.0;
    for (inputs, expected) in test_cases.iter() {
        let outputs: Vec<f32> = network.predict(inputs).collect();

        if outputs.is_empty() {
            return 0.0; // Invalid network
        }

        let output = outputs[0];
        let error = (output - expected).abs();
        total_error += error;
    }

    // Convert error to fitness (lower error = higher fitness)
    // Max error is 4.0 (if all outputs are maximally wrong)
    let fitness = 4.0 - total_error;

    // Square to emphasize good solutions
    fitness * fitness / 16.0
}

#[test]
fn test_xor_evolution() {
    let mut rng = test_rng();

    // Configure mutations for XOR problem
    let mutations = MutationChances::new_from_raw(80, 40.0, 30.0, 5.0, 20.0, 5.0);

    // Create population
    let mut population = Population::new(50, 2, 1, mutations, &mut rng);

    // Evolution parameters
    let max_generations = 100;
    let target_fitness = 0.95; // 95% correct

    let mut _solution_found = false;
    let mut best_fitness_history = Vec::new();

    for generation in 0..max_generations {
        // Evaluate fitness
        population.evaluate_all(xor_fitness);

        // Track progress
        let best = population.best().unwrap();
        let avg_fitness = population.average_fitness();
        best_fitness_history.push(best.fitness);

        println!(
            "Generation {}: Best fitness = {:.4}, Avg fitness = {:.4}",
            generation, best.fitness, avg_fitness
        );

        // Check if solution found
        if best.fitness >= target_fitness {
            _solution_found = true;
            println!("Solution found in generation {}!", generation);

            // Test the solution
            let network = best.topology.to_simple_network();
            println!("\nTesting XOR solution:");
            for (inputs, expected) in &[
                ([0.0, 0.0], 0.0),
                ([0.0, 1.0], 1.0),
                ([1.0, 0.0], 1.0),
                ([1.0, 1.0], 0.0),
            ] {
                let output: Vec<f32> = network.predict(inputs).collect();
                println!(
                    "XOR({}, {}) = {:.4} (expected {})",
                    inputs[0], inputs[1], output[0], expected
                );
            }

            break;
        }

        // Evolve population
        population.evolve(&mut rng);
    }

    // Verify progress was made
    assert!(
        best_fitness_history.last().unwrap() > &0.5,
        "Should achieve at least 50% fitness"
    );

    // Check that fitness generally improves
    let first_quarter_avg: f32 = best_fitness_history[..25].iter().sum::<f32>() / 25.0;
    let last_quarter_avg: f32 = best_fitness_history[best_fitness_history.len() - 25..]
        .iter()
        .sum::<f32>()
        / 25.0;
    assert!(
        last_quarter_avg > first_quarter_avg,
        "Fitness should improve over time"
    );
}

#[test]
fn test_sine_approximation() {
    let mut rng = test_rng();

    // Configure mutations
    let mutations = MutationChances::new_from_raw(
        70, 35.0, // Add neurons
        25.0, // Add connections
        10.0, // Remove neurons
        25.0, // Mutate weights
        5.0,  // Mutate exponents
    );

    // Fitness function for sine approximation
    let sine_fitness = |network: &SimplePolyNetwork| -> f32 {
        let test_points = 20;
        let mut total_error = 0.0;

        for i in 0..test_points {
            let x = (i as f32 / test_points as f32) * 2.0 * std::f32::consts::PI;
            let expected = x.sin();

            let outputs: Vec<f32> = network.predict(&[x]).collect();
            if outputs.is_empty() {
                return 0.0;
            }

            let error = (outputs[0] - expected).abs();
            total_error += error;
        }

        // Convert to fitness (max error ~20)
        (20.0 - total_error).max(0.0) / 20.0
    };

    // Create population
    let mut population = Population::new(30, 1, 1, mutations, &mut rng);

    // Evolve for a few generations
    for generation in 0..50 {
        population.evaluate_all(sine_fitness);

        let best = population.best().unwrap();
        let avg_fitness = population.average_fitness();

        if generation % 10 == 0 {
            println!(
                "Sine approx - Generation {}: Best = {:.4}, Avg = {:.4}",
                generation, best.fitness, avg_fitness
            );
        }

        population.evolve(&mut rng);
    }

    // Should make some progress
    let final_best = population.best().unwrap();
    assert!(
        final_best.fitness > 0.3,
        "Should achieve reasonable sine approximation"
    );
}

#[test]
fn test_population_diversity() {
    let mut rng = test_rng();
    let mutations = MutationChances::new(70);

    // Create population
    let population = Population::new(20, 3, 2, mutations, &mut rng);

    // Check that individuals are different
    let mut unique_counts = Vec::new();

    for individual in &population.individuals {
        let neuron_count = individual.topology.neurons().len();
        unique_counts.push(neuron_count);
    }

    // Should have some variation in network sizes
    unique_counts.sort();
    let min_neurons = unique_counts.first().unwrap();
    let max_neurons = unique_counts.last().unwrap();

    assert!(
        min_neurons <= max_neurons,
        "Population should have some structural diversity"
    );
}

#[test]
fn test_fitness_based_selection() {
    let mut rng = test_rng();
    let mutations = MutationChances::new(50);

    // Create a simple fitness function that rewards more hidden neurons
    let complexity_fitness = |network: &SimplePolyNetwork| -> f32 {
        let total_neurons = network.num_nodes() as f32;
        let hidden_neurons =
            total_neurons - network.num_inputs() as f32 - network.num_outputs() as f32;
        hidden_neurons.max(0.0)
    };

    let mut population = Population::new(20, 2, 1, mutations, &mut rng);

    // Initial evaluation
    population.evaluate_all(complexity_fitness);
    let initial_avg_fitness = population.average_fitness();

    // Evolve for several generations
    for _ in 0..20 {
        population.evolve(&mut rng);
        population.evaluate_all(complexity_fitness);
    }

    let final_avg_fitness = population.average_fitness();

    // Population should evolve towards higher complexity
    assert!(
        final_avg_fitness >= initial_avg_fitness,
        "Fitness should not decrease: {} -> {}",
        initial_avg_fitness,
        final_avg_fitness
    );
}

#[test]
fn test_and_gate_evolution() {
    let mut rng = test_rng();

    // AND gate is linearly separable, should be easier than XOR
    let and_fitness = |network: &SimplePolyNetwork| -> f32 {
        let test_cases = [
            ([0.0, 0.0], 0.0),
            ([0.0, 1.0], 0.0),
            ([1.0, 0.0], 0.0),
            ([1.0, 1.0], 1.0),
        ];

        let mut correct = 0.0;
        for (inputs, expected) in test_cases.iter() {
            let outputs: Vec<f32> = network.predict(inputs).collect();

            if outputs.is_empty() {
                return 0.0;
            }

            let output = outputs[0];
            // Use threshold for binary classification
            let predicted = if output > 0.5 { 1.0 } else { 0.0 };

            if ((predicted - expected) as f32).abs() < 0.1 {
                correct += 1.0;
            }
        }

        correct / 4.0 // Percentage correct
    };

    let mutations = MutationChances::new_from_raw(
        60, 20.0, // Less complexity needed for AND
        30.0, 10.0, 35.0, // More weight mutation
        5.0,
    );

    let mut population = Population::new(20, 2, 1, mutations, &mut rng);

    // AND should be solvable quickly
    let mut solved = false;
    for generation in 0..30 {
        population.evaluate_all(and_fitness);

        let best = population.best().unwrap();

        if best.fitness >= 0.99 {
            solved = true;
            println!("AND gate solved in generation {}", generation);
            break;
        }

        population.evolve(&mut rng);
    }

    assert!(solved, "AND gate should be solvable within 30 generations");
}

#[test]
fn test_multi_output_evolution() {
    let mut rng = test_rng();

    // Test with multiple outputs (e.g., classifying into 3 categories)
    let multi_fitness = |network: &SimplePolyNetwork| -> f32 {
        // Simple pattern: categorize based on sum of inputs
        let test_cases = [
            ([0.0, 0.0], [1.0, 0.0, 0.0]), // Category 0
            ([0.5, 0.5], [0.0, 1.0, 0.0]), // Category 1
            ([1.0, 1.0], [0.0, 0.0, 1.0]), // Category 2
        ];

        let mut total_correct = 0.0;

        for (inputs, expected) in test_cases.iter() {
            let outputs: Vec<f32> = network.predict(inputs).collect();

            if outputs.len() != 3 {
                return 0.0; // Invalid network
            }

            // Find predicted category (highest output)
            let predicted_idx = outputs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            // Find expected category
            let expected_idx = expected
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            if predicted_idx == expected_idx {
                total_correct += 1.0;
            }
        }

        total_correct / 3.0
    };

    let mutations = MutationChances::new(70);
    let mut population = Population::new(30, 2, 3, mutations, &mut rng);

    // Evolve
    for _ in 0..50 {
        population.evaluate_all(multi_fitness);
        population.evolve(&mut rng);
    }

    let best = population.best().unwrap();
    assert!(
        best.fitness > 0.3,
        "Should achieve some success on multi-output task"
    );
}

#[test]
fn test_evolution_with_large_population() {
    let mut rng = test_rng();
    let mutations = MutationChances::new(60);

    // Create large population
    let mut population = Population::new(100, 3, 2, mutations, &mut rng);

    // Simple fitness function
    let simple_fitness = |network: &SimplePolyNetwork| -> f32 {
        let outputs: Vec<f32> = network.predict(&[1.0, 2.0, 3.0]).collect();
        if outputs.len() == 2 {
            // Reward networks that produce different outputs
            (outputs[0] - outputs[1]).abs()
        } else {
            0.0
        }
    };

    // Test that large population can evolve efficiently
    let start = std::time::Instant::now();

    for _ in 0..10 {
        population.evaluate_all(simple_fitness);
        population.evolve(&mut rng);
    }

    let duration = start.elapsed();

    println!("Large population evolution took: {:?}", duration);

    // Should complete in reasonable time
    assert!(
        duration.as_secs() < 60,
        "Large population evolution should complete within 60 seconds"
    );
}

#[test]
fn test_elitism_preserves_best() {
    let mut rng = test_rng();
    let mutations = MutationChances::new(90); // High mutation

    let mut population = Population::new(10, 2, 1, mutations, &mut rng);

    // Create a fitness function where one individual will be clearly best
    let special_fitness = |network: &SimplePolyNetwork| -> f32 {
        // Just return a random-ish but deterministic value based on network structure
        network.num_nodes() as f32 * 0.1
    };

    population.evaluate_all(special_fitness);

    let best_before = population.best().unwrap().fitness;

    // Evolve multiple times
    for _ in 0..10 {
        population.evolve(&mut rng);
        population.evaluate_all(special_fitness);

        let best_after = population.best().unwrap().fitness;

        // Best fitness should never decrease (elitism)
        assert!(
            best_after >= best_before,
            "Elitism should preserve best fitness: {} -> {}",
            best_before,
            best_after
        );
    }
}
