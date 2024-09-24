use crate::{candle_net::expander::Polynomial, prelude::*};
use candle_core::Tensor;
pub struct CandleNetwork {
    pub tensor: Tensor,
}

impl CandleNetwork {
    pub fn from_topology(topology: &NetworkTopology) -> Self {
        for output in topology.neurons().iter().filter_map(|neuron| {
            let neuron = neuron.read().unwrap();
            if neuron.is_output() {
                Some(neuron)
            } else {
                None
            }
        }) {
            //let output_tensor =

            let mut expander = Polynomial::default();

            for input in output.props().unwrap().inputs() {
                let exponent = input.exponent();
                let weight = input.weight();

                expander.handle_operation(exponent, weight);
            }
        }

        todo!()
    }
}
