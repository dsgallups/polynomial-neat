use std::f32::consts::E;

use crate::{candle_net::expander::Polynomial, prelude::*};
use candle_core::Tensor;
use uuid::Uuid;

#[derive(Debug)]
pub struct CandleNetwork {
    neurons: Vec<Polynomial<Uuid>>,
}

impl CandleNetwork {
    pub fn from_topology(topology: &NetworkTopology) -> Self {
        let mut neurons = Vec::with_capacity(topology.neurons().len());

        for output in topology.neurons().iter().filter_map(|neuron| {
            let neuron = neuron.read().unwrap();
            if neuron.is_output() {
                Some(neuron)
            } else {
                None
            }
        }) {
            //let output_tensor =

            let poly = create_polynomial(&output);
            neurons.push(poly)
        }

        Self { neurons }
    }
}

fn create_polynomial(top: &NeuronTopology) -> Polynomial<Uuid> {
    let Some(props) = top.props() else {
        println!("input found");
        //this is an input
        return Polynomial::unit(top.id());
        //todoo
    };

    let mut running_polynomial = Polynomial::default();
    for input in props.inputs() {
        let Some(neuron) = input.neuron() else {
            continue;
        };
        let Ok(neuron) = neuron.read() else {
            panic!("can't read neuron")
        };

        let neuron_polynomial = create_polynomial(&neuron);

        running_polynomial.expand(neuron_polynomial, input.weight(), input.exponent());
    }

    println!("poly: {:?}", running_polynomial);

    running_polynomial
}
