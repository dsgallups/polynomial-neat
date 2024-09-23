use core::fmt;

use crate::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeuronType {
    Input,
    Props(PropsType),
}

impl fmt::Display for NeuronType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input => write!(f, "Input"),
            Self::Props(PropsType::Hidden) => write!(f, "Hidden"),
            Self::Props(PropsType::Output) => write!(f, "Output"),
        }
    }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PropsType {
    Hidden,
    Output,
}

pub struct NeuronProps {
    props_type: PropsType,
    inputs: Vec<NeuronInput>,
}

/// Needs distinction between Hidden and Output since it's a DAG

impl NeuronProps {
    pub fn new(props_type: PropsType, inputs: Vec<NeuronInput>) -> Self {
        Self { props_type, inputs }
    }

    pub fn hidden(inputs: Vec<NeuronInput>) -> Self {
        Self::new(PropsType::Hidden, inputs)
    }
    pub fn output(inputs: Vec<NeuronInput>) -> Self {
        Self::new(PropsType::Output, inputs)
    }

    pub fn props_type(&self) -> PropsType {
        self.props_type
    }

    pub fn inputs(&self) -> &[NeuronInput] {
        self.inputs.as_slice()
    }
}
