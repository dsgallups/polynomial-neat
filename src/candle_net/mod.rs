#![allow(clippy::useless_vec)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
use std::sync::{Arc, RwLock};

use crate::prelude::*;
use candle_core::{Device, Result, Tensor};
use network::CandleNetwork;
use uuid::Uuid;
mod expander;
mod network;

#[cfg(test)]
mod tests;

pub fn scratch() -> Result<()> {
    Ok(())
}
