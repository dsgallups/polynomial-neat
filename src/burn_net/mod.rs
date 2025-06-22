#![allow(clippy::useless_vec)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
use std::sync::{Arc, RwLock};

use crate::prelude::*;
use burn::prelude::*;
use expander::Polynomial;
use fnv::FnvHashMap;
use network::BurnNetwork;
use uuid::Uuid;

mod basis_prime;
mod coeff;
mod expander;
pub mod network;

#[cfg(test)]
mod tests;

fn get_topology_polynomials(topology: &PolyNetworkTopology) -> Vec<Polynomial<Uuid>> {
    let mut neurons = Vec::with_capacity(topology.neurons().len());

    for output in topology.neurons().iter().filter_map(|neuron| {
        let neuron = neuron.read().unwrap();
        if neuron.is_output() {
            Some(neuron)
        } else {
            None
        }
    }) {
        let poly = create_polynomial(&output);
        neurons.push(poly)
    }

    neurons
}

fn create_polynomial(top: &PolyNeuronTopology) -> Polynomial<Uuid> {
    let Some(props) = top.props() else {
        //this is an input
        return Polynomial::unit(top.id());
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

    running_polynomial
}
