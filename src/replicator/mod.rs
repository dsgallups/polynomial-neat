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

        replicants.into_network_topology()
    }
}
#[test]
fn test_simple_replication() {
    todo!();
}
