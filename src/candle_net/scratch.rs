use std::marker::PhantomData;

use super::{expander::Polynomial, CandleNetwork};
use crate::{
    candle_net::{expander::Variable, get_topology_polynomials},
    prelude::*,
};
use candle_core::Tensor;
use fnv::FnvHashMap;
use pretty_assertions::assert_eq;
use uuid::Uuid;

#[test]
fn scratch() {
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
    let output_1 = arc(NeuronTopology::output(
        Uuid::new_v4(),
        vec![InputTopology::downgrade(&hidden_one, 1., 2)],
    ));

    // 2(3x + y)
    //
    // 6x + 2y
    let output_2 = arc(NeuronTopology::output(
        Uuid::new_v4(),
        vec![InputTopology::downgrade(&hidden_one, 2., 1)],
    ));

    let topology = NetworkTopology::from_raw_parts(
        vec![x_n, y_n, hidden_one, output_1, output_2],
        MutationChances::none(),
    );
    let inputs: FnvHashMap<Uuid, usize> = topology
        .neuron_ids()
        .into_iter()
        .enumerate()
        .map(|(v, k)| (k, v))
        .collect();
    let mut output_polynomials = get_topology_polynomials(&topology)
        .into_iter()
        .map(|poly| poly.map_operands(&inputs))
        .collect::<Vec<_>>();

    output_polynomials
        .iter_mut()
        .for_each(|poly| poly.sort_by_exponent(0));

    println!("converted:\n{:#?}", output_polynomials);

    /*
       x^2
       xy
       y^2
       x
       y
    */
    let basis = super::basis_prime::basis_from_poly_list(&output_polynomials);

    println!("basis:\n{:#?}", basis);
    assert_eq!(basis[0], &[Variable::new(1, 2)]); //y^2
    assert_eq!(basis[1], &[Variable::new(0, 1), Variable::new(1, 1)]); //xy
    assert_eq!(basis[2], &[Variable::new(0, 2)]); //x^2
    assert_eq!(basis[3], &[Variable::new(1, 1)]); //x
    assert_eq!(basis[4], &[Variable::new(0, 1)]); //y

    /*
    //  now we need to figure how to create B' for the provided inputs
    //  we will have [1, input1, input2, ...] as the input, multiplied by this vector to get our
    //  basis matrix.
    // if we have 2 inputs, and there are 5 values in the basis matrix, the template will be 5x3.
    let num_cols = inputs.len() + 1;
    let num_rows = basis.len();
    //let basis_prime_template: [[usize; num_cols]; num_rows] = [[0; num_cols]; num_rows];
    let basis_prime_template: BasisPrimeTemplate<usize> =
        BasisPrimeTemplate::new(num_cols, num_rows);*/
}
