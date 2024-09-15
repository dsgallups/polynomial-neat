use std::sync::{Arc, RwLock};

use neuron::{NeuronInputTopology, NeuronTopology};
use rand::Rng;

pub mod activation;
pub mod neuron;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NetworkTopology {
    neurons: Vec<NeuronTopology>,
    // should be between (0..=100). A higher value equates to a higher rate of mutation.
    mutation_rate: u8,
}

impl NetworkTopology {
    pub fn new(
        num_inputs: usize,
        num_outputs: usize,
        mutation_rate: u8,
        rng: &mut impl Rng,
    ) -> Self {
        let neurons = (0..num_inputs)
            .map(|_| NeuronTopology::input())
            .chain((0..num_outputs).map(|_| {
                //a random number of connections to random input neurons;
                let mut chosen_inputs = (0..rng.gen_range(1..num_inputs))
                    .map(|_| {
                        let topology_index = rng.gen_range(0..num_inputs);
                        NeuronInputTopology::new_rand(topology_index, rng)
                    })
                    .collect::<Vec<_>>();

                chosen_inputs.sort_by_key(|top| top.topology_index());
                chosen_inputs.dedup_by_key(|top| top.topology_index());

                NeuronTopology::output_rand(chosen_inputs, &mut rand::thread_rng())
            }))
            .collect::<Vec<_>>();

        Self {
            neurons,
            mutation_rate,
        }
    }

    pub fn from_raw_parts(neurons: Vec<NeuronTopology>, mutation_rate: u8) -> Self {
        Self {
            neurons,
            mutation_rate,
        }
    }

    pub fn replicate(&self, rng: &mut impl Rng) -> Self {
        TopologyReplicator::new(self).replicate(rng)
    }

    pub fn neurons(&self) -> &[NeuronTopology] {
        &self.neurons
    }

    pub fn mutation_rate(&self) -> u8 {
        self.mutation_rate
    }

    pub fn to_network(&self) -> Network {
        Network::from(self)
    }
}

impl From<&NetworkTopology> for Network {
    fn from(value: &NetworkTopology) -> Self {
        let mut neurons: Vec<Arc<RwLock<Neuron>>> = Vec::with_capacity(value.neurons().len());
        let mut input_layer: Vec<Arc<RwLock<Neuron>>> = Vec::new();
        let mut output_layer: Vec<Arc<RwLock<Neuron>>> = Vec::new();

        for neuron_topology in value.neurons() {
            let neuron = neuron_topology.to_neuron(&mut neurons, value.neurons());
            let neuron_read = neuron.read().unwrap();
            if neuron_read.is_input() {
                input_layer.push(Arc::clone(&neuron));
            }
            if neuron_read.is_output() {
                output_layer.push(Arc::clone(&neuron));
            }
        }

        Network::from_raw_parts(neurons, input_layer, output_layer)
    }
}
#[test]
fn test_simple_replication_prediction() {
    use crate::test_utils::simple_neuron_topology;
    let simple_neuron_topology = simple_neuron_topology();

    let simple_network = NetworkTopology::from_raw_parts(simple_neuron_topology, 0); //No mutation occurs, except on the mutation rate.

    let cloned_network = simple_network.replicate(&mut rand::thread_rng());

    let simple_network = simple_network.to_network();
    let cloned_network = cloned_network.to_network();

    let input_value = &[45.];

    let simple_result = simple_network.predict(input_value);
    let cloned_result = cloned_network.predict(input_value);

    for (simple_result, cloned_result) in simple_result.into_iter().zip(cloned_result) {
        assert_eq!(simple_result, cloned_result)
    }
}
