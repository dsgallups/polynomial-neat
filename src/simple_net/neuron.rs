use crate::prelude::*;
use rayon::prelude::*;
use uuid::Uuid;

use super::neuron_type::NeuronType;

pub struct Neuron {
    id: Uuid,
    props: Option<NeuronProps>,
    /// some working value, returned by the result of the activation value.
    activated_value: Option<f32>,
}

impl Neuron {
    pub fn new(id: Uuid, props: Option<NeuronProps>) -> Self {
        Self {
            id,
            props,
            activated_value: None,
        }
    }

    pub fn inputs(&self) -> Option<&[NeuronInput]> {
        self.props.as_ref().map(|props| props.inputs())
    }

    pub fn id(&self) -> Uuid {
        self.id
    }

    pub fn flush_state(&mut self) {
        self.activated_value = None;
    }

    pub fn check_activated(&self) -> Option<f32> {
        self.activated_value
    }

    pub fn neuron_type(&self) -> NeuronType {
        match self.props {
            None => NeuronType::input(),
            Some(ref props) => props.props_type().into(),
        }
    }

    pub fn is_input(&self) -> bool {
        self.neuron_type() == NeuronType::input()
    }
    pub fn is_hidden(&self) -> bool {
        self.neuron_type() == NeuronType::hidden()
    }
    pub fn is_output(&self) -> bool {
        self.neuron_type() == NeuronType::output()
    }

    pub fn activation(&self) -> Option<&(dyn Fn(f32) -> f32 + Send + Sync)> {
        self.props.as_ref().map(|p| p.activation())
    }

    pub fn bias(&self) -> Option<f32> {
        self.props.as_ref().map(|p| p.bias())
    }

    pub fn activate(&mut self) -> f32 {
        if let Some(val) = self.check_activated() {
            return val;
        }
        self.calculate_activation()
    }

    fn calculate_activation(&mut self) -> f32 {
        if self.is_input() {
            return 0.;
        };

        let working_value = self
            .inputs()
            .unwrap()
            .par_iter()
            .map(|input| {
                #[cfg(feature = "debug")]
                {
                    let input_neuron = input.neuron();

                    let in_id = input_neuron.read().unwrap().id();
                    if self.id() == in_id {
                        panic!("Cyclic dependency!");
                    }
                }
                input.get_input_value()
            })
            .sum::<f32>()
            + self.bias().unwrap();
        let result = (self.activation().unwrap())(working_value);

        self.activated_value = Some(result);

        result
    }

    /// used for input nodes.
    pub fn override_state(&mut self, value: f32) {
        self.activated_value = Some(value);
    }
}
