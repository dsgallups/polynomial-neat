#![allow(clippy::useless_vec)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
use std::sync::{Arc, RwLock};

use crate::prelude::*;
use candle_core::{Device, Result, Tensor};
use uuid::Uuid;

mod network;

pub fn scratch() -> Result<()> {
    println!("hello");

    let input = arc(NeuronTopology::input(Uuid::new_v4()));

    let hidden_1 = arc(NeuronTopology::hidden(
        Uuid::new_v4(),
        vec![
            InputTopology::downgrade(&input, 3., 1),
            InputTopology::downgrade(&input, 1., 2),
        ],
    ));

    let hidden_2 = arc(NeuronTopology::hidden(
        Uuid::new_v4(),
        vec![InputTopology::downgrade(&input, 1., 2)],
    ));

    let output = arc(NeuronTopology::output(
        Uuid::new_v4(),
        vec![
            InputTopology::downgrade(&hidden_1, 1., 1),
            InputTopology::downgrade(&hidden_2, 1., 1),
        ],
    ));

    let topology = NetworkTopology::from_raw_parts(
        vec![input, hidden_1, hidden_2, output],
        MutationChances::none(),
    );

    Ok(())
}
