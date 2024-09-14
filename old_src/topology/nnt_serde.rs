use super::*;
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

/// A serializable wrapper for [`NeuronTopology`]. See [`NNTSerde::from`] for conversion.
#[derive(Serialize, Deserialize)]
pub struct NNTSerde<'a, const I: usize, const O: usize> {
    #[serde(with = "BigArray")]
    pub(crate) input_layer: [NeuronTopology<'a>; I],

    pub(crate) hidden_layers: Vec<NeuronTopology<'a>>,

    #[serde(with = "BigArray")]
    pub(crate) output_layer: [NeuronTopology<'a>; O],

    pub(crate) mutation_rate: f32,
    pub(crate) mutation_passes: usize,
}

impl<'a, 'b, const I: usize, const O: usize> From<&NeuralNetworkTopology<'a, I, O>>
    for NNTSerde<'b, I, O>
where
    'a: 'b,
{
    fn from(value: &NeuralNetworkTopology<'a, I, O>) -> Self {
        Self {
            input_layer: value.input_layer.clone(),
            hidden_layers: value.hidden_layers.clone(),
            output_layer: value.output_layer.clone(),
            mutation_rate: value.mutation_rate,
            mutation_passes: value.mutation_passes,
        }
    }
}
