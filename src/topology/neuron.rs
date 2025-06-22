use std::sync::{Arc, RwLock};

use uuid::Uuid;

use crate::prelude::*;

/// This defines a node's topology. What does this mean?
///
/// this node has an identifier.
///
/// Its props are its inputs.
#[derive(Clone, Debug)]
pub struct PolyNeuronTopology {
    id: Uuid,
    neuron_props: Option<PolyNeuronPropsTopology>,
}

impl PolyNeuronTopology {
    /// This creates a topological input node. There are no props
    /// for this type.
    pub fn input(id: Uuid) -> Self {
        Self {
            id,
            neuron_props: None,
        }
    }
    pub fn hidden(id: Uuid, inputs: Vec<PolyInputTopology>) -> Self {
        let neuron_type = PolyNeuronPropsTopology::hidden(inputs);
        Self::new(id, Some(neuron_type))
    }

    pub fn output(id: Uuid, inputs: Vec<PolyInputTopology>) -> Self {
        let neuron_props = PolyNeuronPropsTopology::output(inputs);

        Self::new(id, Some(neuron_props))
    }

    pub fn new(id: Uuid, neuron_props: Option<PolyNeuronPropsTopology>) -> Self {
        Self { id, neuron_props }
    }

    pub fn new_arc(id: Uuid, neuron_props: Option<PolyNeuronPropsTopology>) -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(Self { id, neuron_props }))
    }

    pub fn props(&self) -> Option<&PolyNeuronPropsTopology> {
        self.neuron_props.as_ref()
    }
    pub fn props_mut(&mut self) -> Option<&mut PolyNeuronPropsTopology> {
        self.neuron_props.as_mut()
    }

    /// Note that inputs are reset here.
    pub fn deep_clone(&self) -> Self {
        PolyNeuronTopology {
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

    pub fn to_neuron(&self, neurons: &mut Vec<Arc<RwLock<SimpleNeuron>>>) {
        for neuron in neurons.iter() {
            if neuron.read().unwrap().id() == self.id() {
                return;
            }
        }

        let new_neuron_props = match self.props() {
            Some(topology_props) => {
                let mut new_neuron_inputs = Vec::with_capacity(topology_props.inputs().len());

                for topology_input in topology_props.inputs() {
                    if let Some(topology_input_neuron) = topology_input.neuron() {
                        topology_input_neuron.read().unwrap().to_neuron(neurons);
                        let neuron_in_array = neurons
                            .iter()
                            .find(|n| {
                                n.read().unwrap().id() == topology_input_neuron.read().unwrap().id()
                            })
                            .unwrap();

                        new_neuron_inputs.push(NeuronInput::new(
                            Arc::clone(neuron_in_array),
                            topology_input.weight(),
                            topology_input.exponent(),
                        ));
                    }
                }

                Some(NeuronProps::new(
                    topology_props.props_type(),
                    new_neuron_inputs,
                ))
            }
            None => None,
        };

        let neuron = Arc::new(RwLock::new(SimpleNeuron::new(self.id, new_neuron_props)));
        neurons.push(Arc::clone(&neuron));
    }
}
