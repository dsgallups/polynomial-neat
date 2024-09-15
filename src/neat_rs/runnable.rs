use super::topology::*;

use std::{cell::RefCell, rc::Rc};

/// A runnable, stated Neural Network generated from a [NeuralNetworkTopology].
///
/// Use [`NeuralNetwork::from`] to go from stateles to runnable.
/// Because this has state, you need to run [`NeuralNetwork::flush_state`] between [`NeuralNetwork::predict`] calls.
#[derive(Debug)]
pub struct NeuralNetwork<const I: usize, const O: usize> {
    input_layer: [Rc<RefCell<Neuron>>; I],
    hidden_layers: Vec<Rc<RefCell<Neuron>>>,
    output_layer: [Rc<RefCell<Neuron>>; O],
}

impl<const I: usize, const O: usize> NeuralNetwork<I, O> {
    /// Predicts an output for the given inputs.
    pub fn predict(&self, inputs: [f32; I]) -> [f32; O] {
        for (i, v) in inputs.iter().enumerate() {
            let mut nw = self.input_layer[i].borrow_mut();
            nw.state.value = *v;
            nw.state.processed = true;
        }

        (0..O)
            .map(NeuronLocation::Output)
            .map(|loc| self.process_neuron(loc))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn process_neuron(&self, loc: NeuronLocation) -> f32 {
        let n = self.get_neuron(loc);

        {
            let nr = n.borrow();

            if nr.state.processed {
                return nr.state.value;
            }
        }

        let mut n = n.borrow_mut();

        for (l, w) in n.inputs.clone() {
            n.state.value += self.process_neuron(l) * w;
        }

        n.activate();

        n.state.value
    }

    fn get_neuron(&self, loc: NeuronLocation) -> Rc<RefCell<Neuron>> {
        match loc {
            NeuronLocation::Input(i) => self.input_layer[i].clone(),
            NeuronLocation::Hidden(i) => self.hidden_layers[i].clone(),
            NeuronLocation::Output(i) => self.output_layer[i].clone(),
        }
    }

    /// Flushes the network's state after a [prediction][NeuralNetwork::predict].
    pub fn flush_state(&self) {
        for n in &self.input_layer {
            n.borrow_mut().flush_state();
        }

        for n in &self.hidden_layers {
            n.borrow_mut().flush_state();
        }

        for n in &self.output_layer {
            n.borrow_mut().flush_state();
        }
    }
}

impl<const I: usize, const O: usize> From<&NeuralNetworkTopology<I, O>> for NeuralNetwork<I, O> {
    fn from(value: &NeuralNetworkTopology<I, O>) -> Self {
        let input_layer = value
            .input_layer
            .iter()
            .map(|n| Rc::new(RefCell::new(Neuron::from(&n.read().unwrap().clone()))))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let hidden_layers = value
            .hidden_layers
            .iter()
            .map(|n| Rc::new(RefCell::new(Neuron::from(&n.read().unwrap().clone()))))
            .collect();

        let output_layer = value
            .output_layer
            .iter()
            .map(|n| Rc::new(RefCell::new(Neuron::from(&n.read().unwrap().clone()))))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            input_layer,
            hidden_layers,
            output_layer,
        }
    }
}

/// A state-filled neuron.
#[derive(Clone, Debug)]
pub struct Neuron {
    inputs: Vec<(NeuronLocation, f32)>,
    bias: f32,

    /// The current state of the neuron.
    pub state: NeuronState,

    /// The neuron's activation function
    pub activation: ActivationFn,
}

impl Neuron {
    /// Flushes a neuron's state. Called by [`NeuralNetwork::flush_state`]
    pub fn flush_state(&mut self) {
        self.state.value = self.bias;
    }

    /// Applies the activation function to the neuron
    pub fn activate(&mut self) {
        self.state.value = self.activation.func.activate(self.state.value);
    }
}

impl From<&NeuronTopology> for Neuron {
    fn from(value: &NeuronTopology) -> Self {
        Self {
            inputs: value.inputs.clone(),
            bias: value.bias,
            state: NeuronState {
                value: value.bias,
                ..Default::default()
            },
            activation: value.activation.clone(),
        }
    }
}

/// A state used in [`Neuron`]s for cache.
#[derive(Clone, Debug, Default)]
pub struct NeuronState {
    /// The current value of the neuron. Initialized to a neuron's bias when flushed.
    pub value: f32,

    /// Whether or not [`value`][NeuronState::value] has finished processing.
    pub processed: bool,
}
