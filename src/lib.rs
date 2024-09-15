use std::{
    borrow::BorrowMut,
    ptr::NonNull,
    sync::{Arc, RwLock},
};

mod neat_rs;

pub struct Genome {}

pub struct Network {
    input_layer: Vec<Arc<RwLock<Neuron>>>,
    hidden: Vec<Arc<RwLock<Neuron>>>,
    output_layer: Vec<Arc<RwLock<Neuron>>>,
}

impl Network {
    pub fn predict(&self, inputs: &[f32]) -> Vec<f32> {
        for (i, v) in inputs.iter().enumerate() {
            let Some(nw) = self.input_layer.get(i) else {
                continue;
            };
            // we forcefully hold the lock here since these need to be loaded first.
            let mut nw = nw.write().unwrap();
            nw.state.value = *v;
            nw.state.processed = true;
        }

        todo!();
    }
}

pub struct Neuron {
    /// This is absolutely unsafe and I am going face first into data races and UB. this
    /// will not respect the XOR mutable rules of rust.
    inputs: Vec<Arc<RwLock<Neuron>>>,
    bias: f32,
    state: NeuronState,
    activation: Box<dyn Fn(f32) -> f32 + Send + Sync>,
}

impl Neuron {
    pub fn flush_state(&mut self) {
        self.state.value = self.bias;
    }

    pub fn activate(&mut self) {
        self.state.value = (self.activation)(self.state.value);
    }
}

unsafe impl Send for Neuron {}

unsafe impl Sync for Neuron {}

pub struct NeuronState {
    value: f32,
    processed: bool,
}
