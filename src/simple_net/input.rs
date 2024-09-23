use std::sync::{Arc, RwLock};

use crate::prelude::*;

/// Defines a weight and reference to an input [`Neuron`].
///
/// The topological sibling is [`InputTopology`](crate::topology::neuron::InputTopology);
pub struct NeuronInput {
    neuron: Arc<RwLock<Neuron>>,
    /// weight * (neuron value^exp)
    weight: f32,
    exp: i32,
}

impl NeuronInput {
    pub fn new(neuron: Arc<RwLock<Neuron>>, weight: f32, exp: i32) -> Self {
        Self {
            neuron,
            weight,
            exp,
        }
    }

    /// applies a weight and exponent to the input neuron and returns the result
    pub fn get_input_value(&self) -> f32 {
        let mut neuron = self.neuron.write().unwrap();

        // don't need to activate the neuron since x^0 = 1
        if self.exp == 0 {
            return self.weight;
        }

        let neuron_value = neuron.activate();

        (neuron_value.powi(self.exp)) * self.weight
    }

    #[cfg(feature = "debug")]
    pub fn neuron(&self) -> &Arc<RwLock<Neuron>> {
        &self.neuron
    }
}
