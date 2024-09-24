use std::sync::{Arc, RwLock};

use crate::prelude::*;

pub type NeuronProps = Props<Arc<RwLock<SimpleNeuron>>>;
