use std::sync::Arc;

use rand::Rng;
use uuid::Uuid;

use crate::prelude::*;

#[derive(Clone, Debug)]
pub struct NetworkTopology {
    neurons: Vec<Arc<NeuronTopology>>,
    mutation_chances: MutationChances,
}

impl NetworkTopology {
    pub fn from_raw_parts(
        neurons: Vec<Arc<NeuronTopology>>,
        mutation_chances: MutationChances,
    ) -> Self {
        Self {
            neurons,
            mutation_chances,
        }
    }

    pub fn new(
        num_inputs: usize,
        num_outputs: usize,
        mutation_chances: MutationChances,
        rng: &mut impl Rng,
    ) -> Self {
        let input_neurons = (0..num_inputs)
            .map(|_| Arc::new(NeuronTopology::input(Uuid::new_v4())))
            .collect::<Vec<_>>();

        let output_neurons = (0..num_outputs)
            .map(|_| {
                //a random number of connections to random input neurons;
                let mut chosen_inputs = (0..rng.gen_range(1..input_neurons.len()))
                    .map(|_| {
                        let topology_index = rng.gen_range(0..input_neurons.len());
                        let input = input_neurons.get(topology_index).unwrap();
                        (
                            InputTopology::new_rand(Arc::downgrade(input), &mut rand::thread_rng()),
                            topology_index,
                        )
                    })
                    .collect::<Vec<_>>();

                chosen_inputs.sort_by_key(|(_, i)| *i);
                chosen_inputs.dedup_by_key(|(_, i)| *i);

                let chosen_inputs = chosen_inputs.into_iter().map(|(input, _)| input).collect();

                Arc::new(NeuronTopology::output(Uuid::new_v4(), chosen_inputs))
            })
            .collect::<Vec<_>>();

        let neurons = input_neurons.into_iter().chain(output_neurons).collect();

        Self {
            neurons,
            mutation_chances,
        }
    }

    pub fn new_thoroughly_connected(
        num_inputs: usize,
        num_outputs: usize,
        mutation_chances: MutationChances,
        rng: &mut impl Rng,
    ) -> Self {
        let input_neurons = (0..num_inputs)
            .map(|_| Arc::new(NeuronTopology::input(Uuid::new_v4())))
            .collect::<Vec<_>>();

        let output_neurons = (0..num_outputs)
            .map(|_| {
                //every output neuron is connected to every input neuron

                let chosen_inputs = input_neurons
                    .iter()
                    .map(|input| InputTopology::new_rand(Arc::downgrade(input), rng))
                    .collect::<Vec<_>>();

                Arc::new(NeuronTopology::output(Uuid::new_v4(), chosen_inputs))
            })
            .collect::<Vec<_>>();

        let neurons = input_neurons.into_iter().chain(output_neurons).collect();

        Self {
            neurons,
            mutation_chances,
        }
    }

    pub fn neuron_ids(&self) -> Vec<Uuid> {
        self.neurons.iter().map(|n| n.id()).collect()
    }

    pub fn neurons(&self) -> &[Arc<NeuronTopology>] {
        &self.neurons
    }

    pub fn mutation_chances(&self) -> &MutationChances {
        &self.mutation_chances
    }

    pub fn find_by_id(&self, id: Uuid) -> Option<&Arc<NeuronTopology>> {
        self.neurons.iter().find(|rep| rep.id() == id)
    }

    //#[instrument(skip_all)]
    pub fn replicate(&self, rng: &mut impl Rng) -> NetworkTopology {
        todo!()
    }

    pub fn debug_str(&self) -> String {
        let mut str = String::new();
        for (neuron_index, neuron) in self.neurons.iter().enumerate() {
            str.push_str(&format!(
                "\n(({}) {}[{}]: ",
                neuron_index,
                neuron.id_short(),
                neuron.neuron_type()
            ));
            match neuron.props() {
                Some(props) => {
                    str.push('[');
                    for input in props.inputs() {
                        match input.neuron() {
                            Some(n) => {
                                let loc = self
                                    .neurons
                                    .iter()
                                    .position(|neuron| neuron.id() == n.id())
                                    .unwrap();

                                str.push_str(&format!("({})", loc));
                            }
                            None => str.push_str("(DROPPED)"),
                        }
                    }
                    str.push(']')
                }

                None => {
                    str.push_str("N/A");
                }
            }

            str.push(')');
        }
        str
    }

    //#[instrument(name = "my_span")]
    pub fn to_simple_network(&self) -> SimpleNetwork {
        SimpleNetwork::from_topology(self)
    }
}

#[test]
fn make_simple_network() {
    let input = arc(NeuronTopology::input(Uuid::new_v4()));

    let hidden_1 = arc(NeuronTopology::hidden(
        Uuid::new_v4(),
        vec![
            InputTopology::downgrade(&input, 3., 1),
            InputTopology::downgrade(&input, 1., 2),
        ],
    ));

    let hidden_2 = arc(NeuronTopology::hidden(
        Uuid::new_v4(),
        vec![InputTopology::downgrade(&input, 1., 2)],
    ));

    let output = arc(NeuronTopology::output(
        Uuid::new_v4(),
        vec![
            InputTopology::downgrade(&hidden_1, 1., 1),
            InputTopology::downgrade(&hidden_2, 1., 1),
        ],
    ));

    let topology = NetworkTopology::from_raw_parts(
        vec![input, hidden_1, hidden_2, output],
        MutationChances::none(),
    );

    assert_eq!(topology.neurons().len(), 4);
    assert_eq!(*topology.mutation_chances(), MutationChances::none());
}
