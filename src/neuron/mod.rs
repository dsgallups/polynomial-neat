use rayon::prelude::*;

mod input;
pub use input::*;
use uuid::Uuid;

/// Needs distinction between Hidden and Output since it's a DAG
pub enum NeuronType {
    Input,
    Hidden {
        inputs: Vec<NeuronInput>,
        activation: Box<dyn Fn(f32) -> f32 + Send + Sync>,
        bias: f32,
    },
    Output {
        inputs: Vec<NeuronInput>,
        activation: Box<dyn Fn(f32) -> f32 + Send + Sync>,
        bias: f32,
    },
}

impl NeuronType {
    pub fn hidden(
        inputs: Vec<NeuronInput>,
        activation: Box<dyn Fn(f32) -> f32 + Send + Sync>,
        bias: f32,
    ) -> Self {
        Self::Hidden {
            inputs,
            activation,
            bias,
        }
    }

    pub fn summarize(&self) -> String {
        use NeuronType::*;
        match self {
            Input => "Input".to_string(),
            Hidden {
                inputs,
                activation: _,
                bias,
            } => format!("Hidden: Inputs: {}, Bias: {}", inputs.len(), bias),
            Output {
                inputs,
                activation: _,
                bias,
            } => format!("Output: Inputs: {}, Bias: {}", inputs.len(), bias),
        }
    }

    pub fn output(
        inputs: Vec<NeuronInput>,
        activation: Box<dyn Fn(f32) -> f32 + Send + Sync>,
        bias: f32,
    ) -> Self {
        Self::Output {
            inputs,
            activation,
            bias,
        }
    }

    pub fn input() -> Self {
        Self::Input
    }

    pub fn inputs(&self) -> Option<&[NeuronInput]> {
        use NeuronType::*;
        match &self {
            Input => None,
            Hidden {
                inputs,
                activation: _,
                bias: _,
            }
            | Output {
                inputs,
                activation: _,
                bias: _,
            } => Some(inputs.as_slice()),
        }
    }
    pub fn activation(&self) -> Option<&(dyn Fn(f32) -> f32 + Send + Sync)> {
        use NeuronType::*;
        match &self {
            Input => None,
            Hidden { activation, .. } | Output { activation, .. } => Some(activation),
        }
    }
    pub fn bias(&self) -> Option<f32> {
        use NeuronType::*;
        match &self {
            Input => None,
            Hidden { bias, .. } | Output { bias, .. } => Some(*bias),
        }
    }
}

pub struct Neuron {
    id: Uuid,
    neuron_type: NeuronType,
    /// some working value, returned by the result of the activation value.
    activated_value: Option<f32>,
}

impl Neuron {
    pub fn new(id: Uuid, neuron_type: NeuronType) -> Self {
        Self {
            id,
            neuron_type,
            activated_value: None,
        }
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

    pub fn neuron_type(&self) -> &NeuronType {
        &self.neuron_type
    }

    pub fn activation(&self) -> Option<&(dyn Fn(f32) -> f32 + Send + Sync)> {
        self.neuron_type.activation()
    }

    pub fn is_input(&self) -> bool {
        matches!(self.neuron_type, NeuronType::Input)
    }

    pub fn is_hidden(&self) -> bool {
        matches!(self.neuron_type, NeuronType::Hidden { .. })
    }

    pub fn is_output(&self) -> bool {
        matches!(self.neuron_type, NeuronType::Output { .. })
    }

    pub fn bias(&self) -> Option<f32> {
        self.neuron_type.bias()
    }

    pub fn summarize(&self) -> String {
        format!(
            "Neuron: {}, Type: {}",
            self.id,
            self.neuron_type.summarize()
        )
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
            .neuron_type()
            .inputs()
            .unwrap()
            .par_iter()
            .map(|input| input.get_input_value())
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
