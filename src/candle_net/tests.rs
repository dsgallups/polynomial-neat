use super::{expander::Polynomial, CandleNetwork};
use crate::prelude::*;
use candle_core::{Device, Result, Tensor};
use pretty_assertions::assert_eq;
use uuid::Uuid;
#[test]
pub fn simple_network() {
    let input_id = Uuid::new_v4();

    let input = arc(NeuronTopology::input(input_id));

    let output = arc(NeuronTopology::output(
        Uuid::new_v4(),
        vec![
            InputTopology::downgrade(&input, 1., 1),
            InputTopology::downgrade(&input, 1., 1),
        ],
    ));

    let topology = NetworkTopology::from_raw_parts(vec![input, output], MutationChances::none());

    let polynomials = get_topology_polynomials(&topology);

    assert!(polynomials.len() == 1);
    let poly = &polynomials[0];
    assert_eq!(poly, &Polynomial::new().with_operation(2., input_id, 1));
}

#[test]
pub fn two_input_network() -> Result<()> {
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
        vec![input, hidden_1, hidden_2, output],
        MutationChances::none(),
    );

    let polynomials = get_topology_polynomials(&topology);

    println!("polys: {:#?}", polynomials);

    Ok(())
}

fn get_topology_polynomials(topology: &NetworkTopology) -> Vec<Polynomial<Uuid>> {
    let mut neurons = Vec::with_capacity(topology.neurons().len());

    for output in topology.neurons().iter().filter_map(|neuron| {
        let neuron = neuron.read().unwrap();
        if neuron.is_output() {
            Some(neuron)
        } else {
            None
        }
    }) {
        //let output_tensor =

        let poly = create_polynomial(&output);
        neurons.push(poly)
    }

    neurons
}

fn create_polynomial(top: &NeuronTopology) -> Polynomial<Uuid> {
    let Some(props) = top.props() else {
        //this is an input
        return Polynomial::unit(top.id());
        //todoo
    };

    let mut running_polynomial = Polynomial::default();
    for input in props.inputs() {
        let Some(neuron) = input.neuron() else {
            continue;
        };
        let Ok(neuron) = neuron.read() else {
            panic!("can't read neuron")
        };

        let neuron_polynomial = create_polynomial(&neuron);

        running_polynomial.expand(neuron_polynomial, input.weight(), input.exponent());
    }

    running_polynomial
}
