pub mod neuron;

pub mod network;

pub mod topology;

//pub mod topology;

pub mod prelude {
    pub use super::network::Network;
    pub use super::neuron::{Neuron, NeuronInput, NeuronType};
    pub use super::topology::{
        activation::{Activation, Bias},
        input::InputTopology,
        mutation::{MutationAction, MutationChances, MAX_MUTATIONS},
        network::NetworkTopology,
        neuron::NeuronTopology,
        neuron_type::NeuronTypeTopology,
    };
}

#[cfg(test)]
mod tests;
