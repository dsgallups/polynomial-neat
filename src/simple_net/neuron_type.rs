use crate::prelude::*;

/// Needs distinction between Hidden and Output since it's a DAG
pub enum NeuronType {
    Input,
    Hidden {
        inputs: Vec<NeuronInput>,
        activation: Box<dyn Fn(f32) -> f32 + Send + Sync>,
        bias: f32,
    },
    Output {
        inputs: Vec<NeuronInput>,
        activation: Box<dyn Fn(f32) -> f32 + Send + Sync>,
        bias: f32,
    },
}

impl NeuronType {
    pub fn hidden(
        inputs: Vec<NeuronInput>,
        activation: Box<dyn Fn(f32) -> f32 + Send + Sync>,
        bias: f32,
    ) -> Self {
        Self::Hidden {
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
        Self::Output {
            inputs,
            activation,
            bias,
        }
    }

    pub fn input() -> Self {
        Self::Input
    }

    pub fn inputs(&self) -> Option<&[NeuronInput]> {
        use NeuronType::*;
        match &self {
            Input => None,
            Hidden {
                inputs,
                activation: _,
                bias: _,
            }
            | Output {
                inputs,
                activation: _,
                bias: _,
            } => Some(inputs.as_slice()),
        }
    }
    pub fn activation(&self) -> Option<&(dyn Fn(f32) -> f32 + Send + Sync)> {
        use NeuronType::*;
        match &self {
            Input => None,
            Hidden { activation, .. } | Output { activation, .. } => Some(activation),
        }
    }
    pub fn bias(&self) -> Option<f32> {
        use NeuronType::*;
        match &self {
            Input => None,
            Hidden { bias, .. } | Output { bias, .. } => Some(*bias),
        }
    }
}
