use std::sync::{Arc, RwLock, Weak};

use rand::Rng;

use super::neuron::NeuronTopology;

#[derive(Clone, Debug)]
pub struct InputTopology {
    input: Weak<RwLock<NeuronTopology>>,
    weight: f32,
    exp: i32,
}

impl InputTopology {
    pub fn new(input: Weak<RwLock<NeuronTopology>>, weight: f32, exp: i32) -> Self {
        Self { input, weight, exp }
    }

    pub fn new_rand(input: Weak<RwLock<NeuronTopology>>, rng: &mut impl Rng) -> Self {
        Self {
            input,
            weight: rng.gen_range(-1.0..=1.0),
            exp: rng.gen_range(0..=2),
        }
    }

    pub fn neuron(&self) -> Option<Arc<RwLock<NeuronTopology>>> {
        Weak::upgrade(&self.input)
    }
    pub fn weight(&self) -> f32 {
        self.weight
    }

    pub fn adjust_weight(&mut self, by: f32) {
        self.weight += by;
    }

    pub fn exponent(&self) -> i32 {
        self.exp
    }

    pub fn adjust_exp(&mut self, by: i32) {
        self.exp += by;
    }
}
