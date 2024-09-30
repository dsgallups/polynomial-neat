use std::sync::{Arc, RwLock};

use crate::poly::prelude::*;

pub type NeuronProps = Props<Arc<RwLock<SimpleNeuron>>>;
