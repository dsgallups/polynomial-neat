use std::sync::{Arc, Weak};

use uuid::Uuid;

use crate::prelude::*;

pub struct NeuronReplicants(Vec<Arc<NeuronReplicant>>);

impl NeuronReplicants {
    pub fn with_capacity(cap: usize) -> Self {
        Self(Vec::with_capacity(cap))
    }

    pub fn find_by_id(&self, id: Uuid) -> Option<&Arc<NeuronReplicant>> {
        self.0.iter().find(|rep| rep.id == id)
    }

    pub fn push(&mut self, rep: Arc<NeuronReplicant>) {
        self.0.push(rep);
    }
}

pub struct InputReplicant {
    input: Weak<NeuronReplicant>,
    weight: f32,
}

impl InputReplicant {
    pub fn new(input: Weak<NeuronReplicant>, weight: f32) -> Self {
        Self { input, weight }
    }
}
pub enum NeuronTypeReplicant {
    Input,
    Hidden {
        inputs: Vec<InputReplicant>,
        activation: Activation,
        bias: f32,
    },
    Output {
        inputs: Vec<InputReplicant>,
        activation: Activation,
        bias: f32,
    },
}

impl NeuronTypeReplicant {
    pub fn input() -> Self {
        Self::Input
    }
    pub fn hidden(inputs: Vec<InputReplicant>, activation: Activation, bias: f32) -> Self {
        Self::Hidden {
            inputs,
            activation,
            bias,
        }
    }
    pub fn output(inputs: Vec<InputReplicant>, activation: Activation, bias: f32) -> Self {
        Self::Output {
            inputs,
            activation,
            bias,
        }
    }
}

pub struct NeuronReplicant {
    id: Uuid,
    neuron_type: NeuronTypeReplicant,
}

impl NeuronReplicant {
    pub fn from_topology(
        neuron: &NeuronTopology,
        replicants: &mut NeuronReplicants,
        parent_neurons: &[NeuronTopology],
    ) -> Arc<Self> {
        let id = neuron.id();
        if let Some(replicant) = replicants.find_by_id(id) {
            return Arc::clone(replicant);
        }

        //not found
        match neuron.inputs() {
            None => {
                let val = Arc::new(NeuronReplicant {
                    id,
                    neuron_type: NeuronTypeReplicant::Input,
                });
                replicants.push(Arc::clone(&val));
                val
            }
            Some(inputs) => {
                let mut new_inputs = Vec::with_capacity(inputs.len());
                for input in inputs {
                    let input_neuron_top = parent_neurons.get(input.topology_index()).unwrap();
                    let neuron_replicant = match replicants.find_by_id(input_neuron_top.id()) {
                        Some(replicant) => Arc::downgrade(replicant),
                        None => Arc::downgrade(&NeuronReplicant::from_topology(
                            input_neuron_top,
                            replicants,
                            parent_neurons,
                        )),
                    };
                    let input_replicant = InputReplicant::new(neuron_replicant, input.weight());
                    new_inputs.push(input_replicant);
                }

                let neuron_type = if neuron.is_hidden() {
                    NeuronTypeReplicant::hidden(
                        new_inputs,
                        neuron.activation().unwrap(),
                        neuron.bias().unwrap(),
                    )
                } else {
                    NeuronTypeReplicant::output(
                        new_inputs,
                        neuron.activation().unwrap(),
                        neuron.bias().unwrap(),
                    )
                };
                let val = Arc::new(NeuronReplicant { id, neuron_type });
                replicants.push(Arc::clone(&val));
                val
            }
        }
    }
}
