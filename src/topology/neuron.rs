use rand::Rng;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::activation::Activation;

/// Needs distinction between Hidden and Output since it's a DAG
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NeuronType {
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

impl NeuronType {
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

    pub fn inputs(&self) -> Option<&[InputTopology]> {
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
    pub fn activation(&self) -> Option<Activation> {
        use NeuronType::*;
        match &self {
            Input => None,
            Hidden { activation, .. } | Output { activation, .. } => Some(*activation),
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
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NeuronTopology {
    id: Uuid,
    neuron_type: NeuronType,
}

impl NeuronTopology {
    pub fn new(id: Uuid, neuron_type: NeuronType) -> Self {
        Self { id, neuron_type }
    }
    pub fn input() -> Self {
        Self {
            id: Uuid::new_v4(),
            neuron_type: NeuronType::Input,
        }
    }

    pub fn output_rand(inputs: Vec<InputTopology>, rng: &mut impl Rng) -> Self {
        Self {
            id: Uuid::new_v4(),

            neuron_type: NeuronType::Output {
                inputs,
                activation: Activation::rand(rng),
                bias: rng.gen(),
            },
        }
    }

    pub fn hidden_rand(inputs: Vec<InputTopology>, rng: &mut impl Rng) -> Self {
        Self {
            id: Uuid::new_v4(),
            neuron_type: NeuronType::Hidden {
                inputs,
                activation: Activation::rand(rng),
                bias: rng.gen(),
            },
        }
    }

    pub fn inputs(&self) -> Option<&[InputTopology]> {
        self.neuron_type.inputs()
    }
    pub fn neuron_type(&self) -> &NeuronType {
        &self.neuron_type
    }

    pub fn is_input(&self) -> bool {
        matches!(self.neuron_type, NeuronType::Input)
    }

    pub fn is_hidden(&self) -> bool {
        matches!(self.neuron_type, NeuronType::Hidden { .. })
    }

    pub fn is_output(&self) -> bool {
        matches!(self.neuron_type, NeuronType::Output { .. })
    }

    pub fn id(&self) -> Uuid {
        self.id
    }
    pub fn activation(&self) -> Option<Activation> {
        self.neuron_type.activation()
    }
    pub fn bias(&self) -> Option<f32> {
        self.neuron_type.bias()
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OldNeuronTopology {
    // .0 is the index of the node into the topology. .1 is the weight.
    inputs: Vec<InputTopology>,
    activation: Activation,
    bias: f32,
}

impl OldNeuronTopology {
    pub fn input_rand(rng: &mut impl Rng) -> Self {
        Self {
            inputs: vec![],
            activation: Activation::rand(rng),
            bias: rng.gen(),
        }
    }

    pub fn new_hidden_rand(inputs: Vec<InputTopology>, rng: &mut impl Rng) -> Self {
        Self {
            inputs,
            activation: Activation::rand(rng),
            bias: rng.gen(),
        }
    }

    pub fn new_with_activation_rand(
        inputs: Vec<InputTopology>,
        activation: Activation,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            inputs,
            activation,
            bias: rng.gen(),
        }
    }

    /// Note that the input nodes NEVER use their activation function, but it is best to utilize this just in case.
    ///
    /// TODO(dsgallups): Needs tests for the above comment.
    pub fn input_node_rand(rng: &mut impl Rng) -> Self {
        Self::new_with_activation_rand(vec![], Activation::Linear, rng)
    }

    pub fn activation(&self) -> Activation {
        self.activation
    }
    pub fn inputs(&self) -> &[InputTopology] {
        self.inputs.as_slice()
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct InputTopology {
    topology_index: usize,
    weight: f32,
}

impl InputTopology {
    pub fn new(topology_index: usize, weight: f32) -> Self {
        Self {
            topology_index,
            weight,
        }
    }

    pub fn new_rand(topology_index: usize, rng: &mut impl Rng) -> Self {
        Self {
            topology_index,
            weight: rng.gen_range(-1.0..=1.0),
        }
    }

    pub fn topology_index(&self) -> usize {
        self.topology_index
    }

    pub fn weight(&self) -> f32 {
        self.weight
    }
}
