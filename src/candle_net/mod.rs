#![allow(clippy::useless_vec)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
use crate::prelude::*;
use candle_core::{Device, Result, Tensor};
pub fn scratch() -> Result<()> {
    println!("hello");

    let i0_1 = Input::new((), 3., 1);

    let i0_2 = Input::new((), 1., 2);

    let i1_i = vec![i0_1, i0_2];

    let i2_i = vec![Input::new((), 1., 1)];

    let o_1 = Input::new((), 1., 1);
    let o_2 = Input::new((), 1., 1);

    let o_i = vec![o_1, o_2];

    Ok(())
}
