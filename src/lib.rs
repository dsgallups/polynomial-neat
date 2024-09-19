pub mod neuron;

pub mod network;

pub mod replicator;

//pub mod topology;

#[cfg(test)]
mod test_utils;

pub mod prelude {
    pub use super::network::Network;
    pub use super::neuron::{Neuron, NeuronInput, NeuronType};
    pub use super::replicator::activation::*;
    pub use super::replicator::replicants::*;
    /*pub use super::topology::{
        activation::Activation,
        neuron::{NeuronInputTopology, NeuronTopology, NeuronTopologyType},
        NetworkTopology,
    };*/
}
