use super::get_topology_polynomials;
use crate::{candle_net::expander::Polynomial, prelude::*};
use candle_core::Tensor;
use fnv::FnvHashMap;
use std::f32::consts::E;
use uuid::Uuid;

pub struct CandleNetwork {
    inputs: Vec<f32>,
    outputs: Tensor,
}

impl CandleNetwork {
    pub fn from_topology(topology: &NetworkTopology) -> Self {
        let output_polynomials = get_topology_polynomials(topology);
        let inputs: FnvHashMap<Uuid, usize> = topology
            .neuron_ids()
            .into_iter()
            .enumerate()
            .map(|(v, k)| (k, v))
            .collect();

        let output_polynomials: Vec<Polynomial<usize>> = output_polynomials
            .into_iter()
            .map(|polynomial| polynomial.map_operands(&inputs))
            .collect();

        todo!();
    }
}

#[test]
fn map_inputs_to_outputs() {
    use pretty_assertions::assert_eq;
    let i1_id = Uuid::new_v4();
    let i2_id = Uuid::new_v4();

    println!("Input 1 id: {}\nInput 2 id: {}", i1_id, i2_id);

    let input = arc(NeuronTopology::input(i1_id));
    let input2 = arc(NeuronTopology::input(i2_id));

    let hidden_1 = arc(NeuronTopology::hidden(
        Uuid::new_v4(),
        vec![
            InputTopology::downgrade(&input, 3., 1),
            InputTopology::downgrade(&input, 1., 2),
        ],
    ));

    let hidden_2 = arc(NeuronTopology::hidden(
        Uuid::new_v4(),
        vec![InputTopology::downgrade(&input2, 1., 2)],
    ));

    let hidden_3 = arc(NeuronTopology::output(
        Uuid::new_v4(),
        vec![
            InputTopology::downgrade(&hidden_1, 1., 2),
            InputTopology::downgrade(&hidden_2, 1., 4),
        ],
    ));

    let output = arc(NeuronTopology::output(
        Uuid::new_v4(),
        vec![
            InputTopology::downgrade(&hidden_3, 1., 1),
            InputTopology::downgrade(&hidden_3, 4., 2),
        ],
    ));

    let topology = NetworkTopology::from_raw_parts(
        vec![input, input2, hidden_1, hidden_2, hidden_3, output],
        MutationChances::none(),
    );

    println!("network topology ids: \n{:#?}", topology.neuron_ids());

    let output_polynomials = get_topology_polynomials(&topology);
    let inputs: FnvHashMap<Uuid, usize> = topology
        .neuron_ids()
        .into_iter()
        .enumerate()
        .map(|(v, k)| (k, v))
        .collect();

    let mapped_output_polynomials: Vec<Polynomial<usize>> = output_polynomials
        .clone()
        .into_iter()
        .map(|polynomial| polynomial.map_operands(&inputs))
        .collect();

    for (o, m) in output_polynomials
        .into_iter()
        .zip(mapped_output_polynomials)
    {
        assert_eq!(o.parts().len(), m.parts().len());
        for (op, mp) in o.parts().iter().zip(m.parts()) {
            assert_eq!(op.weight(), mp.weight());
            for (opo, mpo) in op.operands().iter().zip(mp.operands()) {
                assert_eq!(opo.exponent(), mpo.exponent());
                assert_eq!(mpo.var(), inputs.get(opo.var()).unwrap());
            }
        }
    }
}
