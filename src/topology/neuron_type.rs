use std::sync::{RwLock, Weak};

use rand::Rng;

use crate::prelude::*;

pub type PolyNeuronPropsTopology = PolyProps<Weak<RwLock<PolyNeuronTopology>>>;

impl PolyNeuronPropsTopology {
    pub(super) fn set_inputs(&mut self, new_inputs: Vec<PolyInputTopology>) {
        self.inputs = new_inputs;
    }

    /// resets the inputs and copies activation + bias
    pub fn deep_clone(&self) -> Self {
        Self {
            props_type: self.props_type,
            inputs: Vec::with_capacity(self.inputs.len()),
        }
    }

    pub fn add_input(&mut self, input: PolyInputTopology) {
        self.inputs.push(input);
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
    pub fn remove_random_input(&mut self, rng: &mut impl Rng) -> Option<PolyInputTopology> {
        if self.inputs.is_empty() {
            return None;
        }
        let removed = self
            .inputs
            .swap_remove(rng.random_range(0..self.inputs.len()));
        Some(removed)
    }

    pub fn get_random_input_mut(&mut self, rng: &mut impl Rng) -> Option<&mut PolyInputTopology> {
        if self.inputs.is_empty() {
            return None;
        }
        let len = self.inputs.len();
        self.inputs.get_mut(rng.random_range(0..len))
    }
}
