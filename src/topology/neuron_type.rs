use rand::Rng;

use crate::prelude::*;

#[derive(Clone, Debug)]
pub enum NeuronTypeTopology {
    Input,
    Hidden {
        inputs: Vec<InputTopology>,
        activation: Activation,
        bias: f32,
    },
    Output {
        inputs: Vec<InputTopology>,
        activation: Activation,
        bias: f32,
    },
}

impl NeuronTypeTopology {
    pub fn input() -> Self {
        Self::Input
    }
    pub fn hidden(inputs: Vec<InputTopology>, activation: Activation, bias: f32) -> Self {
        Self::Hidden {
            inputs,
            activation,
            bias,
        }
    }
    pub fn output(inputs: Vec<InputTopology>, activation: Activation, bias: f32) -> Self {
        Self::Output {
            inputs,
            activation,
            bias,
        }
    }

    pub(super) fn set_inputs(&mut self, new_inputs: Vec<InputTopology>) {
        use NeuronTypeTopology::*;
        match self {
            Input => panic!("Attempted to set inputs on an input node!"),
            Hidden { inputs, .. } => *inputs = new_inputs,
            Output { inputs, .. } => *inputs = new_inputs,
        }
    }

    /// resets the inputs and copies activation + bias
    pub fn deep_clone(&self) -> Self {
        use NeuronTypeTopology::*;
        match self {
            Input => Input,
            Hidden {
                inputs,
                activation,
                bias,
            } => Hidden {
                inputs: Vec::with_capacity(inputs.len()),
                activation: *activation,
                bias: *bias,
            },
            Output {
                inputs,
                activation,
                bias,
            } => Output {
                inputs: Vec::with_capacity(inputs.len()),
                activation: *activation,
                bias: *bias,
            },
        }
    }

    pub fn add_input(&mut self, input: InputTopology) {
        use NeuronTypeTopology::*;
        match self {
            Input => panic!("Cannot add input to an input node!"),
            Hidden { ref mut inputs, .. } | Output { ref mut inputs, .. } => {
                inputs.push(input);
            }
        }
    }
    pub fn inputs(&self) -> Option<&[InputTopology]> {
        use NeuronTypeTopology::*;
        match self {
            Input => None,
            Hidden { ref inputs, .. } | Output { ref inputs, .. } => Some(inputs),
        }
    }

    // Clears out all inputs whose reference is dropped or match on the provided ids
    pub fn trim_inputs(&mut self, indices: &[usize]) {
        use NeuronTypeTopology::*;
        match self {
            Input => {}
            Hidden { ref mut inputs, .. } | Output { ref mut inputs, .. } => {
                let mut sorted_indices = indices.to_vec();
                sorted_indices.sort_unstable_by(|a, b| b.cmp(a));

                for index in sorted_indices {
                    inputs.remove(index);
                }
            }
        }
    }

    /// Returnes the removed input, if it has inputs.
    pub fn remove_random_input(&mut self, rng: &mut impl Rng) -> Option<InputTopology> {
        use NeuronTypeTopology::*;
        match self {
            Input => None,
            Hidden { inputs, .. } | Output { inputs, .. } => {
                if inputs.is_empty() {
                    return None;
                }
                let removed = inputs.swap_remove(rng.gen_range(0..inputs.len()));
                Some(removed)
            }
        }
    }

    pub fn get_random_input_mut(&mut self, rng: &mut impl Rng) -> Option<&mut InputTopology> {
        use NeuronTypeTopology::*;
        match self {
            Input => None,
            Hidden { inputs, .. } | Output { inputs, .. } => {
                if inputs.is_empty() {
                    return None;
                }
                let len = inputs.len();
                inputs.get_mut(rng.gen_range(0..len))
            }
        }
    }
    pub fn activation(&self) -> Option<Activation> {
        use NeuronTypeTopology::*;
        match self {
            Input => None,
            Hidden { activation, .. } | Output { activation, .. } => Some(*activation),
        }
    }
    pub fn activation_mut(&mut self) -> Option<&mut Activation> {
        use NeuronTypeTopology::*;
        match self {
            Input => None,
            Hidden { activation, .. } | Output { activation, .. } => Some(activation),
        }
    }
    pub fn bias(&self) -> Option<f32> {
        use NeuronTypeTopology::*;
        match self {
            Input => None,
            Hidden { bias, .. } | Output { bias, .. } => Some(*bias),
        }
    }
    pub fn bias_mut(&mut self) -> Option<&mut f32> {
        use NeuronTypeTopology::*;
        match self {
            Input => None,
            Hidden { bias, .. } | Output { bias, .. } => Some(bias),
        }
    }

    pub fn num_inputs(&self) -> usize {
        use NeuronTypeTopology::*;
        match self {
            Input => 0,
            Hidden { inputs, .. } | Output { inputs, .. } => inputs.len(),
        }
    }

    pub fn is_output(&self) -> bool {
        matches!(self, Self::Output { .. })
    }
    pub fn is_input(&self) -> bool {
        matches!(self, Self::Input)
    }
    pub fn is_hidden(&self) -> bool {
        matches!(self, Self::Hidden { .. })
    }
}
