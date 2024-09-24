use std::sync::{Arc, RwLock, Weak};

use crate::prelude::*;

pub type InputTopology = Input<Weak<RwLock<NeuronTopology>>>;

impl InputTopology {
    pub fn neuron(&self) -> Option<Arc<RwLock<NeuronTopology>>> {
        Weak::upgrade(self.input())
    }
}
