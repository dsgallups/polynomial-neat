use std::sync::{Arc, RwLock};

use neuron::{InputTopology, NeuronTopology};
use rand::Rng;

pub mod activation;
pub mod neuron;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{network::Network, neuron::Neuron, prelude::TopologyReplicator};

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
                        InputTopology::new_rand(topology_index, rng)
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
