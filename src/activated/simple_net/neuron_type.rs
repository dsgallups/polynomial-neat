use std::sync::{Arc, RwLock};

use crate::activated::prelude::*;

pub type NeuronProps = PolyProps<Arc<RwLock<SimpleNeuron>>>;
