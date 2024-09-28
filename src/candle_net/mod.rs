#![allow(clippy::useless_vec)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
use std::sync::{Arc, RwLock};

use crate::prelude::*;
use candle_core::{Device, Result, Tensor};
use expander::Polynomial;
use fnv::FnvHashMap;
use network::CandleNetwork;
use uuid::Uuid;
mod basis_prime;
pub mod candle_expander;
mod coeff;
mod expander;
pub mod network;
#[cfg(test)]
mod scratch;

#[cfg(test)]
mod tests;

fn get_topology_polynomials(topology: &NetworkTopology) -> Vec<Polynomial<Uuid>> {
    let mut neurons = Vec::with_capacity(topology.neurons().len());

    for output in topology.neurons().iter().filter_map(|neuron| {
        let neuron = neuron.read().unwrap();
        //println!("get_top_poly, id = {}", neuron.id());
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

    neurons
}

fn create_polynomial(top: &NeuronTopology) -> Polynomial<Uuid> {
    //println!("create_polynomial, id = {}", top.id());
    let Some(props) = top.props() else {
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

    running_polynomial
}
