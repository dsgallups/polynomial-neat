use rand::Rng;
use replicants::{NeuronReplicant, NeuronReplicants};
use uuid::serde::simple;

use crate::{prelude::*, topology::neuron::InputTopology};

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
fn test_simple_replication_properties() {
    let simple_neuron_topology = vec![
        NeuronTopology::input(),
        NeuronTopology::hidden(vec![InputTopology::new(0, 10.)], Activation::Linear, 50.),
        NeuronTopology::output(vec![InputTopology::new(1, 10.)], Activation::Linear, 50.),
    ];

    let simple_network = NetworkTopology::from_raw_parts(simple_neuron_topology, 0); //No mutation occurs, except on the mutation rate.

    let cloned = simple_network.replicate(&mut rand::thread_rng());

    assert_eq!(simple_network.neurons().len(), cloned.neurons().len());

    for (n1, n2) in simple_network.neurons().iter().zip(cloned.neurons()) {
        assert_eq!(n1.is_hidden(), n2.is_hidden());
        assert_eq!(n1.is_input(), n2.is_input());
        assert_eq!(n1.is_output(), n2.is_output());
        assert_eq!(n1.activation(), n2.activation());
        assert_eq!(n1.bias(), n2.bias());
    }
}

#[test]
fn test_simple_replication_prediction() {
    let simple_neuron_topology = vec![
        NeuronTopology::input(),
        NeuronTopology::hidden(vec![InputTopology::new(0, 10.)], Activation::Linear, 50.),
        NeuronTopology::output(vec![InputTopology::new(1, 10.)], Activation::Linear, 50.),
    ];

    let simple_network = NetworkTopology::from_raw_parts(simple_neuron_topology, 0); //No mutation occurs, except on the mutation rate.

    let cloned = simple_network.replicate(&mut rand::thread_rng());

    assert_eq!(simple_network.neurons().len(), cloned.neurons().len());

    let input_value = 45.;
}
