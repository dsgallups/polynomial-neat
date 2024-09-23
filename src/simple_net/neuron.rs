use crate::prelude::*;
use rayon::prelude::*;
use tracing::info;
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
            info!("{} al_act: {}", self.id_short(), val);
            return val;
        }
        info!("-----{} calc", self.id_short());
        let res = self.calculate_activation();
        info!("-----{} re_act: {}", self.id_short(), res);
        res
    }

    fn calculate_activation(&mut self) -> f32 {
        if self.is_input() {
            return 0.;
        };

        let num_inputs = self.inputs().unwrap().len();

        /*
        You may be wondering why this isn't a parallel iterator.
        I have found that if all rayon's threads are blocked,
        then this, even though the "summing" would unblock the rest
        of the threads, cannot complete.
        */
        let result = self
            .inputs()
            .unwrap()
            .par_iter()
            .enumerate()
            .map(|(idx, input)| {
                info!(
                    "{} REQUEST INPUT ({}/{})",
                    self.id_short(),
                    idx,
                    num_inputs - 1
                );
                let res = input.get_input_value(self.id_short(), idx);
                info!(
                    "{} RECEIVED INPUT ({}/{}) ({})",
                    self.id_short(),
                    idx,
                    num_inputs - 1,
                    res
                );
                res
            })
            .sum::<f32>();

        info!("{} RETURNING RESULT FROM INPUTS", self.id_short());

        self.activated_value = Some(result);

        result
    }

    /// used for input nodes.
    pub fn override_state(&mut self, value: f32) {
        self.activated_value = Some(value);
    }
}
