use fnv::FnvHashMap;

use crate::candle_net::get_topology_polynomials;

use super::expander::Polynomial;

pub struct CandleNeuron {
    polynomial: Polynomial<usize>,
}

impl CandleNeuron {
    pub fn new(polynomial: Polynomial<usize>) -> Self {
        todo!();
    }
    pub fn predict(&self, inputs: &[f32]) -> f32 {
        println!("neuron polynomial: {:#?}", self.polynomial);

        todo!();
    }
}
#[test]
fn two_input_one_output() {
    use super::CandleNetwork;
    use crate::prelude::*;
    use pretty_assertions::assert_eq;
    use uuid::Uuid;
    let x_id = Uuid::new_v4();
    let y_id = Uuid::new_v4();

    println!("Input 1 id: {}\nInput 2 id: {}", x_id, y_id);

    let x_n = arc(NeuronTopology::input(x_id));
    let y_n = arc(NeuronTopology::input(y_id));

    let hidden_one = arc(NeuronTopology::hidden(
        Uuid::new_v4(),
        vec![
            InputTopology::downgrade(&x_n, 3., 1),
            InputTopology::downgrade(&y_n, 1., 1),
        ],
    ));

    // (3x + y )^2 =
    // 9x^2 + 6xy + y^2
    let output = arc(NeuronTopology::output(
        Uuid::new_v4(),
        vec![InputTopology::downgrade(&hidden_one, 1., 2)],
    ));

    let topology = NetworkTopology::from_raw_parts(
        vec![x_n, y_n, hidden_one, output],
        MutationChances::none(),
    );

    let network = CandleNetwork::from_topology(&topology);

    //let result = network.predict(&[1., 2.]);
}
