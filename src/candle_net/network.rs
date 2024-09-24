use crate::prelude::*;
use candle_core::Tensor;
pub struct CandleNetwork {
    pub tensor: Tensor,
}

impl CandleNetwork {
    pub fn from_topology(topology: &NetworkTopology) -> Self {
        todo!()
    }
}
