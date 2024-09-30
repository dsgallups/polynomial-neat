use super::{basis_prime::BasisTemplate, coeff::Coefficients, get_topology_polynomials};
use crate::poly::{
    candle_net::{
        basis_prime::basis_from_poly_list,
        expander::{Polynomial, Variable},
    },
    prelude::*,
};
use candle_core::{Device, Result, Tensor};
use fnv::FnvHashMap;
use rayon::iter::{
    IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
    ParallelIterator as _,
};
use std::{env::var, f32::consts::E};
use uuid::Uuid;

pub struct CandleNetwork<'a> {
    coeff_tensor: Coefficients,
    basis_template: BasisTemplate<usize>,
    device: &'a Device,
}

impl<'a> CandleNetwork<'a> {
    pub fn from_topology(topology: &PolyNetworkTopology, device: &'a Device) -> Result<Self> {
        let inputs: FnvHashMap<Uuid, usize> = topology
            .neuron_ids()
            .into_iter()
            .enumerate()
            .map(|(v, k)| (k, v))
            .collect();
        //println!("here 1");

        let mut output_polynomials = get_topology_polynomials(topology)
            .into_par_iter()
            .map(|poly| poly.map_operands(&inputs))
            .collect::<Vec<_>>();
        output_polynomials
            .par_iter_mut()
            .for_each(|poly| poly.sort_by_exponent(0));
        //println!("output polynomials:\n{:?}", output_polynomials);

        let variable_basis = basis_from_poly_list(&output_polynomials);

        //println!("variable basis: {:#?}", variable_basis);
        let basis_template = BasisTemplate::from_raw(variable_basis);
        //println!("basis: {:?}", basis_template);
        //println!("here 4");
        let coeff_tensor = Coefficients::new(&output_polynomials, &basis_template, device)?;
        //println!("coeff:\n{}\n\n", coeff_tensor);
        //println!("here 5");

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

        //println!("basis tensor:\n{}\n\n", basis);
        let result = self.coeff_tensor.inner().matmul(&basis)?;

        //println!("result: {}", result);
        //println!("result work: {}", result.flatten(0, 1)?);
        let res: Vec<f32> = result.flatten(0, 1)?.to_vec1()?;

        Ok(res.into_iter())
    }
}

#[test]
fn candle_scratch() -> Result<()> {
    let x_id = Uuid::new_v4();
    let y_id = Uuid::new_v4();

    println!("Input 1 id: {}\nInput 2 id: {}", x_id, y_id);

    let x_n = arc(PolyNeuronTopology::input(x_id));
    let y_n = arc(PolyNeuronTopology::input(y_id));

    let hidden_one = arc(PolyNeuronTopology::hidden(
        Uuid::new_v4(),
        vec![
            PolyInputTopology::downgrade(&x_n, 3., 1),
            PolyInputTopology::downgrade(&y_n, 1., 1),
        ],
    ));

    // (3x + y )^2 =
    // 9x^2 + 6xy + y^2
    let output_1 = arc(PolyNeuronTopology::output(
        Uuid::new_v4(),
        vec![PolyInputTopology::downgrade(&hidden_one, 1., 2)],
    ));

    // 2(3x + y)
    //
    // 6x + 2y
    let output_2 = arc(PolyNeuronTopology::output(
        Uuid::new_v4(),
        vec![PolyInputTopology::downgrade(&hidden_one, 2., 1)],
    ));

    let topology = PolyNetworkTopology::from_raw_parts(
        vec![x_n, y_n, hidden_one, output_1, output_2],
        MutationChances::none(),
    );

    let candle_net = CandleNetwork::from_topology(&topology, &Device::Cpu)?;

    let res = candle_net.predict(&[3.0, 2.0])?.collect::<Vec<_>>();
    println!("simple_net result: {:?}", res);

    Ok(())
}
#[test]
fn candle_scratch_two() -> Result<()> {
    let topology = PolyNetworkTopology::new(2, 2, MutationChances::none(), &mut rand::thread_rng());

    println!("here 1");
    let candle_net = CandleNetwork::from_topology(&topology, &Device::Cpu)?;

    let res = candle_net.predict(&[3.0, 2.0])?.collect::<Vec<_>>();
    println!("simple_net result: {:?}", res);

    Ok(())
}
