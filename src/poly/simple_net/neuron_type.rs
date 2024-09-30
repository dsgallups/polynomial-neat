use std::sync::{Arc, RwLock};

use crate::poly::prelude::*;

pub type NeuronProps = PolyProps<Arc<RwLock<SimpleNeuron>>>;
