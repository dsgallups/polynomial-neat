use crate::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NeuronType {
    Input,
    Props(PropsType),
}

impl NeuronType {
    pub fn input() -> Self {
        Self::Input
    }
    pub fn hidden() -> Self {
        Self::Props(PropsType::Hidden)
    }
    pub fn output() -> Self {
        Self::Props(PropsType::Output)
    }
}

impl From<PropsType> for NeuronType {
    fn from(value: PropsType) -> Self {
        Self::Props(value)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PropsType {
    Hidden,
    Output,
}

pub struct NeuronProps {
    neuron_type: PropsType,
    inputs: Vec<NeuronInput>,
    activation: Box<dyn Fn(f32) -> f32 + Send + Sync>,
    bias: f32,
}

/// Needs distinction between Hidden and Output since it's a DAG

impl NeuronProps {
    pub fn hidden(
        inputs: Vec<NeuronInput>,
        activation: Box<dyn Fn(f32) -> f32 + Send + Sync>,
        bias: f32,
    ) -> Self {
        Self {
            neuron_type: PropsType::Hidden,
            inputs,
            activation,
            bias,
        }
    }
    pub fn output(
        inputs: Vec<NeuronInput>,
        activation: Box<dyn Fn(f32) -> f32 + Send + Sync>,
        bias: f32,
    ) -> Self {
        Self {
            neuron_type: PropsType::Output,
            inputs,
            activation,
            bias,
        }
    }

    pub fn props_type(&self) -> PropsType {
        self.neuron_type
    }

    pub fn inputs(&self) -> &[NeuronInput] {
        self.inputs.as_slice()
    }
    pub fn activation(&self) -> &(dyn Fn(f32) -> f32 + Send + Sync) {
        &self.activation
    }
    pub fn bias(&self) -> f32 {
        self.bias
    }
}
