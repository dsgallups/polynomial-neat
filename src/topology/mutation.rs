use std::hint::unreachable_unchecked;

use rand::Rng;

pub enum MutationAction {
    SplitConnection,
    Add,
    Remove,
    MutateWeight,
    MutateBias,
    MutateActivationFunction,
}

pub(crate) trait MutationRateExt {
    fn gen_rate(&mut self) -> u8;

    fn gen_mutation_action(&mut self) -> MutationAction;
}

impl<T: Rng> MutationRateExt for T {
    fn gen_rate(&mut self) -> u8 {
        self.gen_range(0..=100)
    }

    fn gen_mutation_action(&mut self) -> MutationAction {
        use MutationAction::*;
        match self.gen_range(0..6) {
            0 => SplitConnection,
            1 => Add,
            2 => Remove,
            3 => MutateWeight,
            4 => MutateBias,
            5 => MutateActivationFunction,
            // Safety: Cannot generate a value more than 5
            _ => unsafe { unreachable_unchecked() },
        }
    }
}

pub const MAX_MUTATIONS: u8 = 200;
