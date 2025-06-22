use super::{BurnNetwork, create_polynomial, expander::Polynomial, get_topology_polynomials};
use crate::{burn_net::expander::PolyComponent, prelude::*};
use burn::backend::NdArray;
use burn::prelude::*;
use fnv::FnvHashMap;
use pretty_assertions::assert_eq;
use uuid::Uuid;

type TestBackend = NdArray;

#[test]
pub fn simple_network() {
    let input_id = Uuid::new_v4();

    let input = arc(PolyNeuronTopology::input(input_id));

    let output = arc(PolyNeuronTopology::output(
        Uuid::new_v4(),
        vec![
            PolyInputTopology::downgrade(&input, 1., 1),
            PolyInputTopology::downgrade(&input, 1., 1),
        ],
    ));

    let topology =
        PolyNetworkTopology::from_raw_parts(vec![input, output], MutationChances::none());

    let polynomials = get_topology_polynomials(&topology);

    assert!(polynomials.len() == 1);
    let poly = &polynomials[0];
    assert_eq!(poly, &Polynomial::new().with_operation(2., input_id, 1));
}

#[test]
pub fn two_input_network() {
    let x = Uuid::new_v4();
    let y = Uuid::new_v4();

    println!("Input 1 id: {}\nInput 2 id: {}", x, y);

    // x
    let input = arc(PolyNeuronTopology::input(x));
    // y
    let input2 = arc(PolyNeuronTopology::input(y));

    // 3x + x^2
    let hidden_1 = arc(PolyNeuronTopology::hidden(
        Uuid::new_v4(),
        vec![
            PolyInputTopology::downgrade(&input, 3., 1),
            PolyInputTopology::downgrade(&input, 1., 2),
        ],
    ));

    // y^2
    let hidden_2 = arc(PolyNeuronTopology::hidden(
        Uuid::new_v4(),
        vec![PolyInputTopology::downgrade(&input2, 1., 2)],
    ));

    // (3x + x^2)^2 + (y^2)^4
    //  x^4 + 6x^3 + 9x^2 + y^8
    let hidden_3 = arc(PolyNeuronTopology::output(
        Uuid::new_v4(),
        vec![
            PolyInputTopology::downgrade(&hidden_1, 1., 2),
            PolyInputTopology::downgrade(&hidden_2, 1., 4),
        ],
    ));

    //  (x^4 + 6x^3 + 9x^2 + y^8) + 4(x^4 + 6x^3 + 9x^2 + y^8)^2
    //
    // 4x^8 + 48x^7 + 216x^6 + 432x^5 + 8x^4y^8 + 325x^4 + 48x^3y^8 +
    //  6x^3 + 72x^2y^8 + 9x^2 + 4y^16 + y^8
    let output = arc(PolyNeuronTopology::output(
        Uuid::new_v4(),
        vec![
            PolyInputTopology::downgrade(&hidden_3, 1., 1),
            PolyInputTopology::downgrade(&hidden_3, 4., 2),
        ],
    ));

    let topology = PolyNetworkTopology::from_raw_parts(
        vec![input, hidden_1, hidden_2, output],
        MutationChances::none(),
    );

    let polynomials = get_topology_polynomials(&topology);
    assert_eq!(polynomials.len(), 1);
    let output_polynomial = polynomials.first().unwrap();

    assert_eq!(output_polynomial.parts().len(), 12);
    let parts = output_polynomial.parts();
    assert_eq!(parts[0], PolyComponent::simple(9., x, 2));
    assert_eq!(parts[1], PolyComponent::simple(6., x, 3));
    assert_eq!(parts[2], PolyComponent::simple(325., x, 4));
    assert_eq!(parts[3], PolyComponent::simple(1., y, 8));
    assert_eq!(parts[4], PolyComponent::simple(432., x, 5));
    assert_eq!(parts[5], PolyComponent::simple(216., x, 6));
    assert_eq!(
        parts[6],
        PolyComponent::new()
            .with_weight(72.)
            .with_operand(x, 2)
            .with_operand(y, 8)
    );

    println!("{:#?}", polynomials);
}

#[test]
fn map_inputs_to_outputs() {
    use pretty_assertions::assert_eq;
    let i1_id = Uuid::new_v4();
    let i2_id = Uuid::new_v4();

    println!("Input 1 id: {}\nInput 2 id: {}", i1_id, i2_id);

    let input = arc(PolyNeuronTopology::input(i1_id));
    let input2 = arc(PolyNeuronTopology::input(i2_id));

    let hidden_1 = arc(PolyNeuronTopology::hidden(
        Uuid::new_v4(),
        vec![
            PolyInputTopology::downgrade(&input, 3., 1),
            PolyInputTopology::downgrade(&input, 1., 2),
        ],
    ));

    let hidden_2 = arc(PolyNeuronTopology::hidden(
        Uuid::new_v4(),
        vec![PolyInputTopology::downgrade(&input2, 1., 2)],
    ));

    let hidden_3 = arc(PolyNeuronTopology::output(
        Uuid::new_v4(),
        vec![
            PolyInputTopology::downgrade(&hidden_1, 1., 2),
            PolyInputTopology::downgrade(&hidden_2, 1., 4),
        ],
    ));

    let output = arc(PolyNeuronTopology::output(
        Uuid::new_v4(),
        vec![
            PolyInputTopology::downgrade(&hidden_3, 1., 1),
            PolyInputTopology::downgrade(&hidden_3, 4., 2),
        ],
    ));

    let topology = PolyNetworkTopology::from_raw_parts(
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

#[test]
fn test_burn_network_functionality() {
    let x_id = Uuid::new_v4();
    let y_id = Uuid::new_v4();

    let x_n = arc(PolyNeuronTopology::input(x_id));
    let y_n = arc(PolyNeuronTopology::input(y_id));

    let hidden_one = arc(PolyNeuronTopology::hidden(
        Uuid::new_v4(),
        vec![
            PolyInputTopology::downgrade(&x_n, 3., 1),
            PolyInputTopology::downgrade(&y_n, 1., 1),
        ],
    ));

    let output_1 = arc(PolyNeuronTopology::output(
        Uuid::new_v4(),
        vec![PolyInputTopology::downgrade(&hidden_one, 1., 2)],
    ));

    let topology = PolyNetworkTopology::from_raw_parts(
        vec![x_n, y_n, hidden_one, output_1],
        MutationChances::none(),
    );

    let device = burn::backend::ndarray::NdArrayDevice::default();
    let burn_net = BurnNetwork::<TestBackend>::from_topology(&topology, device);

    let res = burn_net.predict(&[3.0, 2.0]);
    // (3*3 + 2)^2 = 11^2 = 121
    assert_eq!(res[0], 121.0);
}
