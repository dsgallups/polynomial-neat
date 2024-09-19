use std::hint::unreachable_unchecked;

use rand::Rng;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Activation {
    /// Should only be used on hidden and output nodes
    Sigmoid,
    /// Should only be used on hidden and output nodes
    Relu,
    /// Can be used on all nodes
    Linear,
    /// Should only be used on hidden and output nodes
    Tanh,
}

impl Activation {
    pub fn rand(rng: &mut impl Rng) -> Self {
        use Activation::*;
        match rng.gen_range(0..4) {
            0 => Sigmoid,
            1 => Relu,
            2 => Linear,
            3 => Tanh,
            // Safety: the provided range can only generate values between 0 and 3.
            _ => unsafe { unreachable_unchecked() },
        }
    }

    pub fn as_fn(&self) -> Box<dyn Fn(f32) -> f32 + Send + Sync> {
        use Activation::*;
        match self {
            Sigmoid => Box::new(|n: f32| 1. / (1. + std::f32::consts::E.powf(-n))),
            Relu => Box::new(|n: f32| n.max(0.)),
            Linear => Box::new(|n: f32| n),
            Tanh => Box::new(|n: f32| n.tanh()),
        }
    }
}

pub struct Bias;

impl Bias {
    pub fn rand(rng: &mut impl Rng) -> f32 {
        rng.gen()
    }
}
