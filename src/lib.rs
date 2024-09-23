pub mod candle_net;
pub mod simple_net;
pub mod topology;

pub mod prelude {
    pub use super::simple_net::{
        input::NeuronInput, network::SimpleNetwork, neuron::Neuron, neuron_type::NeuronProps,
    };
    pub use super::topology::{
        activation::{Activation, Bias},
        input::InputTopology,
        mutation::{MutationAction, MutationChances, MAX_MUTATIONS},
        network::NetworkTopology,
        neuron::NeuronTopology,
        neuron_type::NeuronPropsTopology,
    };
}

#[cfg(test)]
mod tests;
