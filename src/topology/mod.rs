use neuron::NeuronTopology;
use rand::Rng;

pub mod activation;
pub mod neuron;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NetworkTopology {
    neurons: Vec<NeuronTopology>,
    input_layer: Vec<usize>,
    output_layer: Vec<usize>,

    mutation_rate: f32,
    mutation_passes: u32,
}

impl NetworkTopology {
    pub fn new(
        num_inputs: usize,
        num_outputs: usize,
        mutation_rate: f32,
        mutation_passes: u32,
        rng: &mut impl Rng,
    ) -> Self {
        let neurons = (0..num_inputs)
            .map(|_| NeuronTopology::input_node_rand(&mut rand::thread_rng()))
            .chain(
                (0..num_outputs).map(|_| NeuronTopology::inputless_rand(&mut rand::thread_rng())),
            )
            .collect::<Vec<_>>();

        Self {
            neurons,
            input_layer: (0..num_inputs).collect(),
            output_layer: (num_inputs..(num_inputs + num_outputs)).collect(),
            mutation_rate,
            mutation_passes,
        }
    }
}

#[test]
fn test_neuron_locations() {
    use activation::Activation;
    // this isn't a flaky test as provided randomness should NOT effect the outcome of the input linear activation function
    // nor correct indexes into the adj array for the input layers.
    let topology = NetworkTopology::new(5, 6, 0., 0, &mut rand::thread_rng());

    for input_i in topology.input_layer.iter() {
        let input_neuron = topology.neurons.get(*input_i).unwrap();
        // always linear
        assert_eq!(input_neuron.activation(), Activation::Linear);
    }

    for output_i in topology.output_layer.iter() {
        topology.neurons.get(*output_i).unwrap();
    }
}
