pub mod candle_net;
pub mod core;
pub mod simple_net;
pub mod topology;
pub mod prelude {
    pub use super::core::{
        activation::{Bias, Exponent},
        input::Input,
        neuron::NeuronInner,
        neuron_type::{NeuronType, Props, PropsType},
    };
    pub use super::simple_net::{
        input::NeuronInput, network::SimpleNetwork, neuron::SimpleNeuron, neuron_type::NeuronProps,
    };
    pub use super::topology::{
        input::InputTopology,
        mutation::{MutationAction, MutationChances, MAX_MUTATIONS},
        network::NetworkTopology,
        neuron::NeuronTopology,
        neuron_type::NeuronPropsTopology,
    };
}

#[cfg(test)]
mod tests;
