use crate::prelude::*;

pub(crate) fn simple_neuron_topology() -> Vec<NeuronTopology> {
    vec![
        NeuronTopology::input(),
        NeuronTopology::hidden(
            vec![NeuronInputTopology::new(0, 10.)],
            Activation::Linear,
            50.,
        ),
        NeuronTopology::output(
            vec![NeuronInputTopology::new(1, 10.)],
            Activation::Linear,
            50.,
        ),
    ]
}
