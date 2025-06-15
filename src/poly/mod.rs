//! A polynomial network

pub mod burn_net;
// pub mod candle_net;  // Commented out - replaced by burn_net
pub mod core;
pub mod simple_net;
pub mod topology;
pub mod prelude {
    pub use super::core::{
        activation::{Bias, Exponent},
        input::PolyInput,
        neuron::PolyNeuronInner,
        neuron_type::{NeuronType, PolyProps, PropsType},
    };
    pub use super::simple_net::{
        input::NeuronInput, network::SimplePolyNetwork, neuron::SimpleNeuron,
        neuron_type::NeuronProps,
    };
    pub use super::topology::{
        input::PolyInputTopology,
        mutation::{MAX_MUTATIONS, MutationAction, MutationChances},
        network::PolyNetworkTopology,
        neuron::PolyNeuronTopology,
        neuron_type::PolyNeuronPropsTopology,
    };
    #[cfg(test)]
    pub(crate) use crate::test_utils::arc;
}

#[cfg(test)]
mod tests;
