use std::sync::{Arc, RwLock};

use super::Neuron;

/// Defines a weight and reference to an input [`Neuron`].
///
/// The topological sibling is [`InputTopology`](crate::topology::neuron::InputTopology);
pub struct NeuronInput {
    neuron: Arc<RwLock<Neuron>>,
    /// neuron value * weight
    weight: f32,
}

impl NeuronInput {
    pub fn new(neuron: Arc<RwLock<Neuron>>, weight: f32) -> Self {
        Self { neuron, weight }
    }

    /// applies a weight to the input neuron and returns the result
    pub fn get_input_value(&self) -> f32 {
        let mut neuron = self.neuron.write().unwrap();

        let neuron_value = neuron.activate();

        neuron_value * self.weight
    }

    #[cfg(feature = "debug")]
    pub fn neuron(&self) -> &Arc<RwLock<Neuron>> {
        &self.neuron
    }
}
