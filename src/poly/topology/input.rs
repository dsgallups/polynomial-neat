use std::sync::{Arc, RwLock, Weak};

use crate::poly::prelude::*;

pub type InputTopology = Input<Weak<RwLock<NeuronTopology>>>;

impl InputTopology {
    pub fn neuron(&self) -> Option<Arc<RwLock<NeuronTopology>>> {
        Weak::upgrade(self.input())
    }

    pub fn downgrade(input: &Arc<RwLock<NeuronTopology>>, weight: f32, exp: i32) -> Self {
        Self::new(Arc::downgrade(input), weight, exp)
    }
}
