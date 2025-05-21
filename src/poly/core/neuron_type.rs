use std::fmt;

use crate::poly::prelude::*;

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

#[derive(Clone, Debug)]
pub struct Props<I> {
    pub(crate) props_type: PropsType,
    pub(crate) inputs: Vec<Input<I>>,
}

impl<I> Props<I> {
    pub fn new(props_type: PropsType, inputs: Vec<Input<I>>) -> Self {
        Self { props_type, inputs }
    }
    pub fn hidden(inputs: Vec<Input<I>>) -> Self {
        Self::new(PropsType::Hidden, inputs)
    }
    pub fn output(inputs: Vec<Input<I>>) -> Self {
        Self::new(PropsType::Output, inputs)
    }

    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    pub fn inputs(&self) -> &[Input<I>] {
        self.inputs.as_slice()
    }

    pub fn props_type(&self) -> PropsType {
        self.props_type
    }
}
