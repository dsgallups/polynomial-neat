use std::sync::{Arc, RwLock};

use rayon::iter::{
    IndexedParallelIterator as _, IntoParallelRefIterator as _, ParallelIterator as _,
};
use tracing::{error, info};

use crate::prelude::*;

pub struct SimpleNetwork {
    // contains all neurons
    neurons: Vec<Arc<RwLock<Neuron>>>,
    // contains the input neurons. cloned arc of neurons in neurons
    input_layer: Vec<Arc<RwLock<Neuron>>>,
    // contains the output neurons. cloned arc of neurons in neurons
    output_layer: Vec<Arc<RwLock<Neuron>>>,
}

impl SimpleNetwork {
    /// Flushes the previous state of the network and calculates given new inputs.
    pub fn predict(&self, inputs: &[f32]) -> impl Iterator<Item = f32> {
        println!("{}", self.debug_str());
        // reset all states first
        self.neurons.iter().enumerate().for_each(|(i, neuron)| {
            info!("ROOT WRITEFLUSH({})", i);
            let mut neuron = neuron.write().unwrap();
            neuron.flush_state();
            /*if neuron.is_input() {
                neuron.override_state(inputs[input_i]);
                input_i += 1;
            }*/

            info!("ROOT DROPFLUSH({}, {})", i, neuron.id_short());
            drop(neuron);
        });
        /*inputs.iter().enumerate().for_each(|(index, value)| {
            info!("ROOT WRITEINPUT({})", index);
            let Some(nw) = self.input_layer.get(index) else {
                panic!("couldn't flush i {}", index);
            };
            let mut nw = nw.write().unwrap();
            nw.override_state(*value);

            info!(
                "ROOT DROPWRITEINPUT({}, {}, value = {})",
                index,
                nw.id_short(),
                value
            );
            drop(nw);
        });*/

        info!("Now iterating through outputs!");
        for (i, neuron) in self.neurons.iter().enumerate() {
            match neuron.try_write() {
                Ok(_) => info!("{} not blocked", i),
                Err(e) => info!("{} blocked: {:?}", i, e),
            }
        }

        let outputs =
            self.output_layer
                .iter()
                .enumerate()
                .fold(Vec::new(), |mut values, (index, neuron)| {
                    let result = {
                        info!("ROOT WRITEOUTPUT({})", index);
                        let mut neuron = neuron.write().unwrap();
                        info!("ROOT WRITEOUTPUT({}, {}, locked)", index, neuron.id_short());
                        for (i, neuron_2) in self.neurons.iter().enumerate() {
                            match neuron_2.try_write() {
                                Ok(neuron_2) => {
                                    info!(
                                        "with lock({}), {}({:?}) not blocked",
                                        neuron.id_short(),
                                        i,
                                        neuron_2.id_short()
                                    )
                                }
                                Err(e) => {
                                    let neuron_2_read =
                                        neuron_2.try_read().ok().map(|n2| n2.id_short());
                                    info!(
                                        "with lock({}), {}({:?}) blocked: {:?}",
                                        neuron.id_short(),
                                        i,
                                        neuron_2_read,
                                        e
                                    )
                                }
                            }
                        }
                        let result = neuron.activate();
                        info!(
                            "ROOT WRITEDROPOUTPUT({}, {}, output calculated)",
                            index,
                            neuron.id_short()
                        );
                        result
                    };

                    values.push(result);

                    values
                });
        //.collect_vec_list();

        outputs.into_iter()
        //.flat_map(|outer_vec| outer_vec.into_iter())
        //.flat_map(|inner_vec| inner_vec.into_iter())
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
                        let n = input.neuron().read().unwrap();

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
                        let n = input.neuron().read().unwrap();

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
                        let n = input.neuron().read().unwrap();

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
}
