use rand::Rng;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::activation::Activation;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NeuronTopology {
    // .0 is the index of the node into the topology. .1 is the weight.
    inputs: Vec<InputTopology>,
    bias: f32,
    activation: Activation,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct InputTopology {
    topology_index: usize,
    weight: f32,
}

impl NeuronTopology {
    pub fn inputless_rand(rng: &mut impl Rng) -> Self {
        Self {
            inputs: vec![],
            activation: Activation::rand(rng),
            bias: rng.gen(),
        }
    }

    pub fn new_rand(inputs: Vec<InputTopology>, rng: &mut impl Rng) -> Self {
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
}
