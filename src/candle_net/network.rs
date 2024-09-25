use super::{basis_prime::BasisTemplate, coeff::Coefficients, get_topology_polynomials};
use crate::{
    candle_net::{
        basis_prime::basis_from_poly_list,
        expander::{Polynomial, Variable},
    },
    prelude::*,
};
use candle_core::{Device, Result, Tensor};
use fnv::FnvHashMap;
use std::f32::consts::E;
use uuid::Uuid;

pub struct CandleNetwork<'a> {
    coeff_tensor: Coefficients,
    basis_template: BasisTemplate<usize>,
    device: &'a Device,
}

impl<'a> CandleNetwork<'a> {
    pub fn from_topology(topology: &NetworkTopology, device: &'a Device) -> Result<Self> {
        let inputs: FnvHashMap<Uuid, usize> = topology
            .neuron_ids()
            .into_iter()
            .enumerate()
            .map(|(v, k)| (k, v))
            .collect();

        let output_polynomials = get_topology_polynomials(topology)
            .into_iter()
            .map(|poly| {
                let mut new = poly.map_operands(&inputs);
                new.sort_by_exponent(0);
                new
            })
            .collect::<Vec<_>>();

        let variable_basis = basis_from_poly_list(&output_polynomials);

        let basis_template = BasisTemplate::new(&output_polynomials);
        let coeff_tensor = Coefficients::new(&output_polynomials, &basis_template, device)?;

        Ok(Self {
            coeff_tensor,
            basis_template,
            device,
        })
    }

    pub fn predict(&self, inputs: &[f32]) -> Result<impl Iterator<Item = f32>> {
        let basis = self
            .basis_template
            .make_tensor(inputs.iter().enumerate().map(|(p, v)| (p, *v)), self.device)?;

        let result = self.coeff_tensor.inner().matmul(&basis)?;
        let res: Vec<f32> = result.to_vec1()?;

        Ok(res.into_iter())
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
