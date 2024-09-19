pub mod neuron;

pub mod network;

pub mod topology;

//pub mod topology;

#[cfg(test)]
mod test_utils;

pub mod prelude {
    pub use super::network::Network;
    pub use super::neuron::{Neuron, NeuronInput, NeuronType};
    pub(crate) use super::topology::mutation::MutationRateExt;
    pub use super::topology::{
        activation::{Activation, Bias},
        input::InputTopology,
        mutation::{MutationAction, MAX_MUTATIONS},
        network::NetworkTopology,
        neuron::NeuronTopology,
        neuron_type::NeuronTypeTopology,
    };
}
