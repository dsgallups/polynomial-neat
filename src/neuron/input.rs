use std::sync::{Arc, RwLock};

use super::Neuron;

pub struct NeuronInput {
    neuron: Arc<RwLock<Neuron>>,
    /// neuron value * weight
    weight: f32,
}

impl NeuronInput {
    /// applies a weight to the input neuron and returns the result
    pub fn get_input_value(&self) -> f32 {
        let mut neuron = self.neuron.write().unwrap();

        let neuron_value = neuron.activate();

        neuron_value * self.weight
    }
}
