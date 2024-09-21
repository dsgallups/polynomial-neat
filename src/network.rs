use std::sync::{Arc, RwLock};

use rayon::iter::{
    IndexedParallelIterator as _, IntoParallelRefIterator as _, ParallelIterator as _,
};
use tracing::info;

use crate::prelude::*;

pub struct Network {
    // contains all neurons
    neurons: Vec<Arc<RwLock<Neuron>>>,
    // contains the input neurons. cloned arc of neurons in neurons
    input_layer: Vec<Arc<RwLock<Neuron>>>,
    // contains the output neurons. cloned arc of neurons in neurons
    output_layer: Vec<Arc<RwLock<Neuron>>>,
}

impl Network {
    /// Flushes the previous state of the network and calculates given new inputs.
    pub fn predict(&self, inputs: &[f32]) -> impl Iterator<Item = f32> {
        // reset all states first
        info!("resetting states");
        self.neurons.par_iter().for_each(|neuron| {
            let mut neuron = neuron.write().unwrap();
            neuron.flush_state();
        });

        info!("setting inputs");
        inputs.par_iter().enumerate().for_each(|(index, value)| {
            let Some(nw) = self.input_layer.get(index) else {
                return;
            };
            let mut nw = nw.write().unwrap();
            nw.override_state(*value);
        });

        info!("activating neurons");
        let outputs = self
            .output_layer
            .par_iter()
            .fold(Vec::new, |mut values, neuron| {
                let mut neuron = neuron.write().unwrap();

                values.push(neuron.activate());

                values
            })
            .collect_vec_list();

        info!("returning outputs");

        outputs
            .into_iter()
            .flat_map(|outer_vec| outer_vec.into_iter())
            .flat_map(|inner_vec| inner_vec.into_iter())
    }

    pub fn from_raw_parts(
        neurons: Vec<Arc<RwLock<Neuron>>>,
        input_layer: Vec<Arc<RwLock<Neuron>>>,
        output_layer: Vec<Arc<RwLock<Neuron>>>,
    ) -> Self {
        Self {
            neurons,
            input_layer,
            output_layer,
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.neurons.len()
    }
    pub fn num_inputs(&self) -> usize {
        self.input_layer.len()
    }
    pub fn num_outputs(&self) -> usize {
        self.output_layer.len()
    }
}
