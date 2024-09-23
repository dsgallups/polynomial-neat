use std::sync::{Arc, RwLock};

use tracing::info;

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
    pub fn get_input_value(&self, caller: String, idx: usize) -> f32 {
        // don't need to activate the neuron since x^0 = 1
        if self.exp == 0 {
            return self.weight;
        }

        let cached = {
            let val = self.neuron().read().unwrap().check_activated();
            if let Some(val) = val {
                Some(val.powi(self.exp) * self.weight)
            } else {
                None
            }
        };
        info!("Getting input value for {}({})", caller, idx);
        if let Some(cached) = cached {
            info!("--{}({}) Cached and returning", caller, idx);
            cached.powi(self.exp) * self.weight
        } else {
            info!("--{}({}) Noncached. locking", caller, idx);

            if let Err(e) = self.neuron.try_write() {
                let read_lock = self.neuron.read().unwrap();
                info!(
                    "Couldn't try_write for {}({}): {:?}. Neuron id: {}. Status of neuron in question: {:?}",
                    caller,
                    idx,
                    e,
                    read_lock.id_short(),
                    read_lock.check_activated()
                );
            }
            let neuron_value = self.neuron.write().unwrap().activate();
            info!("--{}({}) now holds write lock", caller, idx);
            neuron_value.powi(self.exp) * self.weight
        }
    }

    pub fn neuron(&self) -> &Arc<RwLock<Neuron>> {
        &self.neuron
    }
}
