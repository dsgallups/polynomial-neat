use std::sync::{Arc, RwLock};

use uuid::Uuid;

use crate::{prelude::*, simple_net::neuron_type::NeuronType};

#[derive(Clone, Debug)]
pub struct NeuronTopology {
    id: Uuid,
    neuron_props: Option<NeuronPropsTopology>,
}

impl NeuronTopology {
    pub fn input(id: Uuid) -> Self {
        Self {
            id,
            neuron_props: None,
        }
    }
    pub fn hidden(id: Uuid, inputs: Vec<InputTopology>) -> Self {
        let neuron_type = NeuronPropsTopology::hidden(inputs);
        Self::new(id, Some(neuron_type))
    }

    pub fn output(id: Uuid, inputs: Vec<InputTopology>) -> Self {
        let neuron_props = NeuronPropsTopology::output(inputs);

        Self::new(id, Some(neuron_props))
    }

    pub fn new(id: Uuid, neuron_props: Option<NeuronPropsTopology>) -> Self {
        Self { id, neuron_props }
    }

    pub fn new_arc(id: Uuid, neuron_props: Option<NeuronPropsTopology>) -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(Self { id, neuron_props }))
    }

    pub fn props(&self) -> Option<&NeuronPropsTopology> {
        self.neuron_props.as_ref()
    }
    pub fn props_mut(&mut self) -> Option<&mut NeuronPropsTopology> {
        self.neuron_props.as_mut()
    }

    /// Note that inputs are reset here.
    pub fn deep_clone(&self) -> Self {
        NeuronTopology {
            id: Uuid::new_v4(),
            neuron_props: self.neuron_props.as_ref().map(|props| props.deep_clone()),
        }
    }

    pub fn id(&self) -> Uuid {
        self.id
    }

    pub fn id_short(&self) -> String {
        let str = self.id.to_string();
        str[0..6].to_string()
    }

    pub fn neuron_type(&self) -> NeuronType {
        match self.neuron_props {
            None => NeuronType::input(),
            Some(ref p) => p.props_type().into(),
        }
    }

    pub fn is_output(&self) -> bool {
        self.neuron_type() == NeuronType::output()
    }

    pub fn is_hidden(&self) -> bool {
        self.neuron_type() == NeuronType::hidden()
    }
    pub fn is_input(&self) -> bool {
        self.neuron_type() == NeuronType::input()
    }

    pub fn to_neuron(&self, neurons: &mut Vec<Arc<RwLock<Neuron>>>) -> Arc<RwLock<Neuron>> {
        for neuron in neurons.iter() {
            if neuron.read().unwrap().id() == self.id() {
                return Arc::clone(neuron);
            }
        }

        let neuron_props = match self.props() {
            Some(props) => {
                let mut new_inputs = Vec::with_capacity(props.inputs().len());

                for input in props.inputs() {
                    if let Some(input_neuron) = input.neuron() {
                        let neuron = input_neuron.read().unwrap().to_neuron(neurons);
                        new_inputs.push(NeuronInput::new(neuron, input.weight(), input.exponent()));
                    }
                }

                Some(NeuronProps::new(props.props_type(), new_inputs))
            }
            None => None,
        };

        let neuron = Arc::new(RwLock::new(Neuron::new(self.id, neuron_props)));
        neurons.push(Arc::clone(&neuron));

        neuron
    }
}
