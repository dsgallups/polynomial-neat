use rayon::prelude::*;

mod input;
pub use input::*;

pub struct Neuron {
    /// This is absolutely unsafe and I am going face first into data races and UB. this
    /// will not respect the XOR mutable rules of rust.
    inputs: Vec<NeuronInput>,
    bias: f32,

    /// some working value, returned by the result of the activation value.
    activated_value: Option<f32>,
    activation: Box<dyn Fn(f32) -> f32 + Send + Sync>,
}

impl Neuron {
    pub fn flush_state(&mut self) {
        self.activated_value = None;
    }

    pub fn check_activated(&self) -> Option<f32> {
        self.activated_value
    }

    pub fn activate(&mut self) -> f32 {
        if let Some(val) = self.check_activated() {
            return val;
        }
        self.calculate_activation()
    }

    fn calculate_activation(&mut self) -> f32 {
        let working_value = self
            .inputs
            .par_iter()
            .map(|input| input.get_input_value())
            .sum::<f32>()
            + self.bias;
        let result = (self.activation)(working_value);

        self.activated_value = Some(result);

        result
    }

    /// used for input nodes.
    pub fn override_state(&mut self, value: f32) {
        self.activated_value = Some(value);
    }
}
