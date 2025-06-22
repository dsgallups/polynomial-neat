use std::sync::{Arc, RwLock};

use crate::prelude::*;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator as _, ParallelIterator as _};
use uuid::Uuid;

pub struct SimpleNeuron {
    id: Uuid,
    props: Option<PolyProps<Arc<RwLock<SimpleNeuron>>>>,
    /// some working value, returned by the result of the activation value.
    activated_value: Option<f32>,
}

impl SimpleNeuron {
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

    pub fn props(&self) -> Option<&NeuronProps> {
        self.props.as_ref()
    }

    pub fn id_short(&self) -> String {
        let str = self.id.to_string();
        str[0..6].to_string()
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

        /*
           Deeply nested like this will block all threads on rayon.
           we cannot use rayon here, but an async implementation *could* work.
        */
        let result = self
            .inputs()
            .unwrap()
            .par_iter()
            .by_uniform_blocks(1)
            .map(|input| input.get_input_value())
            .sum::<f32>();

        self.activated_value = Some(result);

        result
    }

    /// used for input nodes.
    pub fn override_state(&mut self, value: f32) {
        self.activated_value = Some(value);
    }
}
