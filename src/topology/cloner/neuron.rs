use std::sync::{RwLock, Weak};

use crate::prelude::*;

pub type CloningNeuronProps = Props<Weak<RwLock<NeuronReplicator>>>;
pub type CloningInput = Input<Weak<RwLock<NeuronReplicator>>>;

#[derive(Clone, Debug)]
pub struct NeuronReplicator {
    neuron_props: Option<CloningNeuronProps>,
}

impl NeuronReplicator {
    pub fn input() -> Self {
        Self { neuron_props: None }
    }
    pub fn hidden(inputs: Vec<CloningInput>) -> Self {
        let neuron_type = CloningNeuronProps::hidden(inputs);
        Self::new(Some(neuron_type))
    }

    pub fn output(inputs: Vec<CloningInput>) -> Self {
        let neuron_props = CloningNeuronProps::output(inputs);

        Self::new(Some(neuron_props))
    }

    pub fn new(neuron_props: Option<CloningNeuronProps>) -> Self {
        Self { neuron_props }
    }

    pub fn props(&self) -> Option<&CloningNeuronProps> {
        self.neuron_props.as_ref()
    }
    pub fn props_mut(&mut self) -> Option<&mut CloningNeuronProps> {
        self.neuron_props.as_mut()
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
}
