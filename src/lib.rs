pub mod neuron;

pub mod network;

pub mod replicator;

pub mod topology;

pub mod prelude {
    pub use super::network::Network;
    pub use super::neuron::{Neuron, NeuronInput};
    pub use super::replicator::TopologyReplicator;
    pub use super::topology::{
        activation::Activation,
        neuron::{NeuronTopology, NeuronType},
        NetworkTopology,
    };
}
