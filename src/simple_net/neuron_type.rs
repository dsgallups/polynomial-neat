use std::sync::{Arc, RwLock};

use crate::prelude::*;

pub type NeuronProps = PolyProps<Arc<RwLock<SimpleNeuron>>>;
