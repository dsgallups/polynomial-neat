use std::sync::{Arc, Weak};

use crate::prelude::*;

pub type InputTopology = Input<Weak<NeuronTopology>>;

impl InputTopology {
    pub fn neuron(&self) -> Option<Arc<NeuronTopology>> {
        Weak::upgrade(self.input())
    }

    pub fn downgrade(input: &Arc<NeuronTopology>, weight: f32, exp: i32) -> Self {
        Self::new(Arc::downgrade(input), weight, exp)
    }
}
