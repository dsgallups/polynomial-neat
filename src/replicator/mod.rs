use rand::Rng;
use replicants::{NeuronReplicant, NeuronReplicants};

use crate::prelude::*;

mod replicants;

pub struct TopologyReplicator<'a> {
    parent_neurons: &'a [NeuronTopology],
    parent_mutation_rate: u8,
}

impl<'a> TopologyReplicator<'a> {
    pub fn new(topology: &'a NetworkTopology) -> Self {
        Self {
            parent_neurons: topology.neurons(),
            parent_mutation_rate: topology.mutation_rate(),
        }
    }

    pub fn replicate(self, rng: &mut impl Rng) -> NetworkTopology {
        let mut replicants = NeuronReplicants::with_capacity(self.parent_neurons.len());

        for neuron in self.parent_neurons.iter() {
            NeuronReplicant::from_topology(neuron, &mut replicants, self.parent_neurons);
        }

        replicants.mutate(self.parent_mutation_rate, rng);

        let neuron_topology = replicants.into_neuron_topology();

        let new_mutation_rate = if rng.gen_bool(0.5) {
            self.parent_mutation_rate.saturating_add(1)
        } else {
            self.parent_mutation_rate.saturating_sub(1)
        };

        NetworkTopology::from_raw_parts(neuron_topology, new_mutation_rate)
    }
}
#[test]
fn test_simple_replication() {
    todo!();
}
