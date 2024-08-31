use crate::topology::*;

use rayon::prelude::*;
use std::sync::RwLock;

/// A runnable, stated Neural Network generated from a [NeuralNetworkTopology]. Use [`NeuralNetwork::from`] to go from stateles to runnable.
/// Because this has state, you need to run [`NeuralNetwork::flush_state`] between [`NeuralNetwork::predict`] calls.
/// Parallelized version of the [`NeuralNetwork`] struct.
#[derive(Debug)]
pub struct NeuralNetwork<'f, const I: usize, const O: usize> {
    input_layer: [RwLock<Neuron<'f>>; I],
    hidden_layers: Vec<RwLock<Neuron<'f>>>,
    output_layer: [RwLock<Neuron<'f>>; O],
}

impl<'f, const I: usize, const O: usize> NeuralNetwork<'f, I, O> {
    /// Parallelized prediction of outputs from inputs.
    pub fn predict(&self, inputs: [f32; I]) -> [f32; O] {
        inputs.par_iter().enumerate().for_each(|(i, v)| {
            let mut nw = self.input_layer[i].write().unwrap();
            nw.state.value = *v;
            nw.state.processed = true;
        });

        (0..O)
            .map(NeuronLocation::Output)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|loc| self.process_neuron(loc))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn process_neuron(&self, loc: NeuronLocation) -> f32 {
        let n = self.get_neuron(loc);

        {
            let nr = n.read().unwrap();

            if nr.state.processed {
                return nr.state.value;
            }
        }

        let val: f32 = n
            .read()
            .unwrap()
            .inputs
            .par_iter()
            .map(|&(n2, w)| {
                let processed = self.process_neuron(n2);
                processed * w
            })
            .sum();

        let mut nw = n.write().unwrap();
        nw.state.value += val;
        nw.activate();

        nw.state.value
    }

    fn get_neuron(&self, loc: NeuronLocation) -> &RwLock<Neuron<'f>> {
        match loc {
            NeuronLocation::Input(i) => &self.input_layer[i],
            NeuronLocation::Hidden(i) => &self.hidden_layers[i],
            NeuronLocation::Output(i) => &self.output_layer[i],
        }
    }

    /// Flushes the network's state after a [prediction][NeuralNetwork::predict].
    pub fn flush_state(&self) {
        self.input_layer
            .par_iter()
            .for_each(|n| n.write().unwrap().flush_state());

        self.hidden_layers
            .par_iter()
            .for_each(|n| n.write().unwrap().flush_state());

        self.output_layer
            .par_iter()
            .for_each(|n| n.write().unwrap().flush_state());
    }
}

impl<'a, const I: usize, const O: usize> From<&NeuralNetworkTopology<'a, I, O>>
    for NeuralNetwork<'a, I, O>
{
    fn from(value: &NeuralNetworkTopology<'a, I, O>) -> Self {
        let input_layer = value
            .input_layer
            .iter()
            .map(|n| RwLock::new(Neuron::from(n)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let hidden_layers = value
            .hidden_layers
            .iter()
            .map(|n| RwLock::new(Neuron::from(n)))
            .collect();

        let output_layer = value
            .output_layer
            .iter()
            .map(|n| RwLock::new(Neuron::from(n)))
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
pub struct Neuron<'f> {
    inputs: Vec<(NeuronLocation, f32)>,
    bias: f32,

    /// The current state of the neuron.
    pub state: NeuronState,

    /// The neuron's activation function
    pub activation: ActivationFn<'f>,
}

impl<'f> Neuron<'f> {
    /// Flushes a neuron's state. Called by [`NeuralNetwork::flush_state`]
    pub fn flush_state(&mut self) {
        self.state.value = self.bias;
    }

    /// Applies the activation function to the neuron
    pub fn activate(&mut self) {
        self.state.value = self.activation.func.activate(self.state.value);
    }
}

impl<'f> From<&NeuronTopology<'f>> for Neuron<'f> {
    fn from(value: &NeuronTopology<'f>) -> Self {
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

/// A blanket trait for iterators meant to help with interpreting the output of a [`NeuralNetwork`]
#[cfg(feature = "max-index")]
pub trait MaxIndex<T: PartialOrd> {
    /// Retrieves the index of the max value.
    fn max_index(self) -> usize;
}

#[cfg(feature = "max-index")]
impl<I: Iterator<Item = T>, T: PartialOrd> MaxIndex<T> for I {
    // slow and lazy implementation but it works (will prob optimize in the future)
    fn max_index(self) -> usize {
        self.enumerate()
            .max_by(|(_, v), (_, v2)| v.partial_cmp(v2).unwrap())
            .unwrap()
            .0
    }
}
