pub mod neat_rs;

pub mod neuron;

pub mod network;

pub mod topology;

pub mod prelude {
    pub use super::network::Network;
    pub use super::neuron::{Neuron, NeuronInput};
}
