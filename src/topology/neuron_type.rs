use rand::Rng;

use crate::{prelude::*, simple_net::neuron_type::PropsType};

#[derive(Clone, Debug)]
pub struct NeuronPropsTopology {
    props_type: PropsType,
    inputs: Vec<InputTopology>,
}

impl NeuronPropsTopology {
    pub fn hidden(inputs: Vec<InputTopology>) -> Self {
        Self {
            props_type: PropsType::Hidden,
            inputs,
        }
    }
    pub fn output(inputs: Vec<InputTopology>) -> Self {
        Self {
            props_type: PropsType::Output,
            inputs,
        }
    }

    pub(super) fn set_inputs(&mut self, new_inputs: Vec<InputTopology>) {
        self.inputs = new_inputs;
    }

    /// resets the inputs and copies activation + bias
    pub fn deep_clone(&self) -> Self {
        Self {
            props_type: self.props_type,
            inputs: Vec::with_capacity(self.inputs.len()),
        }
    }

    pub fn add_input(&mut self, input: InputTopology) {
        self.inputs.push(input);
    }
    pub fn inputs(&self) -> &[InputTopology] {
        self.inputs.as_slice()
    }

    // Clears out all inputs whose reference is dropped or match on the provided ids
    pub fn trim_inputs(&mut self, indices: &[usize]) {
        let mut sorted_indices = indices.to_vec();
        sorted_indices.sort_unstable_by(|a, b| b.cmp(a));

        for index in sorted_indices {
            self.inputs.remove(index);
        }
    }

    /// Returnes the removed input, if it has inputs.
    pub fn remove_random_input(&mut self, rng: &mut impl Rng) -> Option<InputTopology> {
        if self.inputs.is_empty() {
            return None;
        }
        let removed = self.inputs.swap_remove(rng.gen_range(0..self.inputs.len()));
        Some(removed)
    }

    pub fn get_random_input_mut(&mut self, rng: &mut impl Rng) -> Option<&mut InputTopology> {
        if self.inputs.is_empty() {
            return None;
        }
        let len = self.inputs.len();
        self.inputs.get_mut(rng.gen_range(0..len))
    }

    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    pub fn props_type(&self) -> PropsType {
        self.props_type
    }
}
