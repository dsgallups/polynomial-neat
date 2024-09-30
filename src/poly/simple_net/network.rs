use std::sync::{Arc, RwLock};

use rayon::iter::{IndexedParallelIterator as _, IntoParallelRefIterator, ParallelIterator as _};

use crate::poly::prelude::*;

pub struct SimplePolyNetwork {
    // contains all neurons
    neurons: Vec<Arc<RwLock<SimpleNeuron>>>,
    // contains the input neurons. cloned arc of neurons in neurons
    input_layer: Vec<Arc<RwLock<SimpleNeuron>>>,
    // contains the output neurons. cloned arc of neurons in neurons
    output_layer: Vec<Arc<RwLock<SimpleNeuron>>>,
}

impl SimplePolyNetwork {
    /// Flushes the previous state of the network and calculates given new inputs.
    pub fn predict(&self, inputs: &[f32]) -> impl Iterator<Item = f32> {
        // reset all states first
        self.neurons.par_iter().for_each(|neuron| {
            let mut neuron = neuron.write().unwrap();
            neuron.flush_state();
        });
        inputs.par_iter().enumerate().for_each(|(index, value)| {
            let Some(nw) = self.input_layer.get(index) else {
                //sim
                return;
                //panic!("couldn't flush i {}", index);
            };
            let mut nw = nw.write().unwrap();
            nw.override_state(*value);
        });

        let outputs = self
            .output_layer
            .par_iter()
            .fold(Vec::new, |mut values, neuron| {
                let mut neuron = neuron.write().unwrap();

                values.push(neuron.activate());

                values
            })
            .collect_vec_list();

        outputs
            .into_iter()
            .flat_map(|outer_vec| outer_vec.into_iter())
            .flat_map(|inner_vec| inner_vec.into_iter())
    }

    pub fn from_raw_parts(
        neurons: Vec<Arc<RwLock<SimpleNeuron>>>,
        input_layer: Vec<Arc<RwLock<SimpleNeuron>>>,
        output_layer: Vec<Arc<RwLock<SimpleNeuron>>>,
    ) -> Self {
        Self {
            neurons,
            input_layer,
            output_layer,
        }
    }

    pub fn summarize(&self) -> String {
        format!(
            "Network with \n{} total nodes\n{} input nodes\n{} output nodes",
            self.num_nodes(),
            self.num_inputs(),
            self.num_outputs()
        )
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

    pub fn debug_str(&self) -> String {
        let mut str = "neurons: \n".to_string();
        for (neuron_index, neuron) in self.neurons.iter().enumerate() {
            let neuron = neuron.read().unwrap();
            str.push_str(&format!(
                "\n(({}) {}[{}]: ",
                neuron_index,
                neuron.id_short(),
                neuron.neuron_type()
            ));
            match neuron.props() {
                Some(props) => {
                    str.push('[');
                    for input in props.inputs() {
                        let n = input.input().read().unwrap();

                        let loc = self
                            .neurons
                            .iter()
                            .position(|neuron| neuron.read().unwrap().id() == n.id())
                            .unwrap();

                        str.push_str(&format!("({})", loc));
                    }
                    str.push(']')
                }

                None => {
                    str.push_str("N/A");
                }
            }

            str.push(')');
        }

        str.push_str("\n\ninput_layer:");

        for (neuron_index, neuron) in self.input_layer.iter().enumerate() {
            let neuron = neuron.read().unwrap();
            str.push_str(&format!(
                "\n(({}) {}[{}]: ",
                neuron_index,
                neuron.id_short(),
                neuron.neuron_type()
            ));
            match neuron.props() {
                Some(props) => {
                    str.push('[');
                    for input in props.inputs() {
                        let n = input.input().read().unwrap();

                        let loc = self
                            .neurons
                            .iter()
                            .position(|neuron| neuron.read().unwrap().id() == n.id())
                            .unwrap();

                        str.push_str(&format!("({})", loc));
                    }
                    str.push(']')
                }

                None => {
                    str.push_str("N/A");
                }
            }

            str.push(')');
        }

        str.push_str("\n\noutput layer:");

        for (neuron_index, neuron) in self.output_layer.iter().enumerate() {
            let neuron = neuron.read().unwrap();
            str.push_str(&format!(
                "\n(({}) {}[{}]: ",
                neuron_index,
                neuron.id_short(),
                neuron.neuron_type()
            ));
            match neuron.props() {
                Some(props) => {
                    str.push('[');
                    for input in props.inputs() {
                        let n = input.input().read().unwrap();

                        let loc = self
                            .neurons
                            .iter()
                            .position(|neuron| neuron.read().unwrap().id() == n.id())
                            .unwrap();

                        str.push_str(&format!("({})", loc));
                    }
                    str.push(']')
                }

                None => {
                    str.push_str("N/A");
                }
            }

            str.push(')');
        }

        str
    }

    pub fn from_topology(topology: &PolyNetworkTopology) -> Self {
        let mut neurons: Vec<Arc<RwLock<SimpleNeuron>>> =
            Vec::with_capacity(topology.neurons().len());
        let mut input_layer: Vec<Arc<RwLock<SimpleNeuron>>> = Vec::new();
        let mut output_layer: Vec<Arc<RwLock<SimpleNeuron>>> = Vec::new();

        for neuron_replicant in topology.neurons() {
            let neuron = neuron_replicant.read().unwrap();

            neuron.to_neuron(&mut neurons);
            let neuron = neurons
                .iter()
                .find(|n| n.read().unwrap().id() == neuron.id())
                .unwrap();

            let neuron_read = neuron.read().unwrap();

            if neuron_read.is_input() {
                input_layer.push(Arc::clone(neuron));
            }
            if neuron_read.is_output() {
                output_layer.push(Arc::clone(neuron));
            }
        }

        SimplePolyNetwork::from_raw_parts(neurons, input_layer, output_layer)
    }
}
