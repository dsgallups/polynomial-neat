use std::{
    f32,
    sync::{Arc, RwLock},
};

use crate::prelude::*;

pub type NeuronInput = PolyInput<Arc<RwLock<SimpleNeuron>>>;

impl NeuronInput {
    /// applies a weight and exponent to the input neuron and returns the result
    pub fn get_input_value(&self) -> f32 {
        // don't need to activate the neuron since x^0 = 1
        if self.exponent() == 0 {
            return self.weight();
        }

        //note that this can totally return inf if the value is zero. This is intentional.

        if let Some(cached) = {
            self.input()
                .read()
                .unwrap()
                .check_activated()
                .map(|val| val.powi(self.exponent()) * self.weight())
        } {
            return cached;
        }

        let neuron_value = self.input().write().unwrap().activate();

        // if result.is_infinite() || result.is_nan() {
        //     println!(
        //         "DEBUG: Fresh calculation produced inf/nan: neuron_value={}, exp={}, weight={}, result={}",
        //         neuron_value,
        //         self.exponent(),
        //         self.weight(),
        //         result
        //     );
        // }
        // result

        neuron_value.powi(self.exponent()) * self.weight()
    }
}

#[test]
fn ensure_inf_possible() {
    use uuid::Uuid;
    let neuron = Arc::new(RwLock::new(SimpleNeuron::new(
        Uuid::new_v4(),
        Some(NeuronProps::output(vec![PolyInput::new(
            Arc::new(RwLock::new(SimpleNeuron::new(Uuid::new_v4(), None))),
            0.,
            1,
        )])),
    )));
    let neuron_input = PolyInput::new(neuron, 1., -1);

    let value = neuron_input.get_input_value();
    assert_eq!(value, f32::INFINITY);
}
#[test]
fn zero_to_zero_power() {
    use uuid::Uuid;
    let neuron = Arc::new(RwLock::new(SimpleNeuron::new(
        Uuid::new_v4(),
        Some(NeuronProps::output(vec![PolyInput::new(
            Arc::new(RwLock::new(SimpleNeuron::new(Uuid::new_v4(), None))),
            0.,
            1,
        )])),
    )));
    let neuron_input = PolyInput::new(neuron, 1., 0);

    let value = neuron_input.get_input_value();
    assert_eq!(value, 1.);
}
