use crate::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NetworkCheckpoint {
    mutation_rate: u8,
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NeuronCheckpoint {
    id: Uuid,
    neuron_type: 
}