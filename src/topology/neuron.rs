use std::sync::{Arc, RwLock};

use rand::Rng;
use uuid::Uuid;

use crate::prelude::*;

#[derive(Clone, Debug)]
pub struct NeuronTopology {
    id: Uuid,
    neuron_type: NeuronTypeTopology,
}

impl NeuronTopology {
    pub fn input(id: Uuid) -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(Self {
            id,
            neuron_type: NeuronTypeTopology::input(),
        }))
    }

    pub fn output_rand(inputs: Vec<InputTopology>, rng: &mut impl Rng) -> Arc<RwLock<Self>> {
        let neuron_type =
            NeuronTypeTopology::output(inputs, Activation::rand(rng), Bias::rand(rng));

        Arc::new(RwLock::new(Self {
            id: Uuid::new_v4(),
            neuron_type,
        }))
    }

    pub fn new(id: Uuid, neuron_type: NeuronTypeTopology) -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(Self { id, neuron_type }))
    }

    pub(super) fn set_inputs(&mut self, new_inputs: Vec<InputTopology>) {
        self.neuron_type.set_inputs(new_inputs);
    }

    /// Note that inputs are reset here.
    pub fn deep_clone(&self) -> Self {
        NeuronTopology {
            id: Uuid::new_v4(),
            neuron_type: self.neuron_type.deep_clone(),
        }
    }

    pub fn hidden_rand(inputs: Vec<InputTopology>, rng: &mut impl Rng) -> Arc<RwLock<Self>> {
        let id = Uuid::new_v4();
        let neuron_type =
            NeuronTypeTopology::hidden(inputs, Activation::rand(rng), Bias::rand(rng));
        Self::new(id, neuron_type)
    }

    pub fn get_random_input_mut(&mut self, rng: &mut impl Rng) -> Option<&mut InputTopology> {
        self.neuron_type.get_random_input_mut(rng)
    }

    pub fn activation(&self) -> Option<Activation> {
        self.neuron_type.activation()
    }

    pub fn activation_mut(&mut self) -> Option<&mut Activation> {
        self.neuron_type.activation_mut()
    }
    pub fn bias(&self) -> Option<f32> {
        self.neuron_type.bias()
    }
    pub fn bias_mut(&mut self) -> Option<&mut f32> {
        self.neuron_type.bias_mut()
    }
    pub fn num_inputs(&self) -> usize {
        self.neuron_type.num_inputs()
    }

    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Returnes the removed input, if it has inputs.
    pub fn remove_random_input(&mut self, rng: &mut impl Rng) -> Option<InputTopology> {
        self.neuron_type.remove_random_input(rng)
    }

    pub fn add_input(&mut self, input: InputTopology) {
        self.neuron_type.add_input(input)
    }

    pub fn inputs(&self) -> Option<&[InputTopology]> {
        self.neuron_type.inputs()
    }

    pub fn is_output(&self) -> bool {
        self.neuron_type.is_output()
    }
    pub fn is_input(&self) -> bool {
        self.neuron_type.is_input()
    }
    pub fn is_hidden(&self) -> bool {
        self.neuron_type.is_hidden()
    }

    pub fn trim_inputs(&mut self, ids: &[usize]) {
        self.neuron_type.trim_inputs(ids)
    }

    pub fn to_neuron(
        &self,
        neurons: &mut Vec<Arc<RwLock<Neuron>>>,
        _replicants: &[Arc<RwLock<NeuronTopology>>],
    ) -> Arc<RwLock<Neuron>> {
        for neuron in neurons.iter() {
            if neuron.read().unwrap().id() == self.id() {
                return Arc::clone(neuron);
            }
        }

        let neuron_type = if let Some(inputs) = self.inputs() {
            let mut new_inputs = Vec::with_capacity(inputs.len());
            for input in inputs {
                if let Some(input_neuron) = input.neuron() {
                    let neuron = input_neuron.read().unwrap().to_neuron(neurons, _replicants);
                    new_inputs.push(NeuronInput::new(neuron, input.weight()));
                }
            }

            if self.is_hidden() {
                NeuronProps::Hidden {
                    inputs: new_inputs,
                    activation: self.activation().unwrap().as_fn(),
                    bias: self.bias().unwrap(),
                }
            } else {
                NeuronProps::Output {
                    inputs: new_inputs,
                    activation: self.activation().unwrap().as_fn(),
                    bias: self.bias().unwrap(),
                }
            }
        } else {
            NeuronProps::Input
        };

        let neuron = Neuron::new(self.id, neuron_type);

        Arc::new(RwLock::new(neuron))
    }
}
