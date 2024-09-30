pub mod candle_net;
pub mod core;
pub mod simple_net;
mod test_utils;
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
    #[cfg(test)]
    pub(crate) use super::test_utils::arc;
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
