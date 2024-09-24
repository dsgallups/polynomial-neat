use std::sync::{Arc, RwLock};

use crate::prelude::*;

pub type NeuronInput = Input<Arc<RwLock<Neuron>>>;

impl NeuronInput {
    /// applies a weight and exponent to the input neuron and returns the result
    pub fn get_input_value(&self) -> f32 {
        // don't need to activate the neuron since x^0 = 1
        if self.exponent() == 0 {
            return self.weight();
        }

        if let Some(cached) = {
            self.input()
                .read()
                .unwrap()
                .check_activated()
                .map(|val| val.powi(self.exponent()) * self.weight())
        } {
            return cached;
        }

        let neuron_value = self.input().write().unwrap().activate();
        neuron_value.powi(self.exponent()) * self.weight()
    }
}
