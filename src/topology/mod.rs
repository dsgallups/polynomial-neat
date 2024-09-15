use neuron::{InputTopology, NeuronTopology};
use rand::Rng;

pub mod activation;
pub mod neuron;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::TopologyReplicator;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NetworkTopology {
    neurons: Vec<NeuronTopology>,
    mutation_rate: f32,
    mutation_passes: u32,
}

impl NetworkTopology {
    pub fn new(
        num_inputs: usize,
        num_outputs: usize,
        mutation_rate: f32,
        mutation_passes: u32,
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
            mutation_passes,
        }
    }

    pub fn replicate(&self, rng: &mut impl Rng) -> Self {
        TopologyReplicator::new(self).replicate(rng)
    }

    pub fn neurons(&self) -> &[NeuronTopology] {
        &self.neurons
    }

    pub fn mutation_rate(&self) -> f32 {
        self.mutation_rate
    }
    pub fn mutation_passes(&self) -> u32 {
        self.mutation_passes
    }
}
