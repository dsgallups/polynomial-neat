pub mod neuron;

pub mod network;

pub mod replicator;

pub mod topology;

#[cfg(test)]
mod test_utils;

pub mod prelude {
    pub use super::network::Network;
    pub use super::neuron::{Neuron, NeuronInput, NeuronType};
    pub use super::replicator::TopologyReplicator;
    pub use super::topology::{
        activation::Activation,
        neuron::{NeuronInputTopology, NeuronTopology, NeuronTopologyType},
        NetworkTopology,
    };
}

#[test]
fn test_stack_overflow() {
    use crate::prelude::*;
    let some_topology = NetworkTopology::new(20, 7, 80, &mut rand::thread_rng());
    let mut children = Vec::new();
    for _ in (0..=50) {
        let res = some_topology.replicate(&mut rand::thread_rng());
        children.push(res);
    }

    for child in children {
        //println!("{}", child.mutation_rate())
    }

    //println!("children: {:?}", children);
}
