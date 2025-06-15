use super::{basis_prime::BasisTemplate, coeff::Coefficients, get_topology_polynomials};
use crate::poly::{
    burn_net::{
        basis_prime::basis_from_poly_list,
        expander::{Polynomial, Variable},
    },
    prelude::*,
};
use burn::prelude::*;
use fnv::FnvHashMap;
use rayon::iter::{
    IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
    ParallelIterator as _,
};
use std::f32::consts::E;
use uuid::Uuid;

pub struct BurnNetwork<B: Backend> {
    coeff_tensor: Coefficients<B>,
    basis_template: BasisTemplate<usize>,
    device: B::Device,
}

impl<B: Backend> BurnNetwork<B> {
    pub fn from_topology(topology: &PolyNetworkTopology, device: B::Device) -> Self {
        let inputs: FnvHashMap<Uuid, usize> = topology
            .neuron_ids()
            .into_iter()
            .enumerate()
            .map(|(v, k)| (k, v))
            .collect();

        let mut output_polynomials = get_topology_polynomials(topology)
            .into_par_iter()
            .map(|poly| poly.map_operands(&inputs))
            .collect::<Vec<_>>();
        output_polynomials
            .par_iter_mut()
            .for_each(|poly| poly.sort_by_exponent(0));

        let variable_basis = basis_from_poly_list(&output_polynomials);

        let basis_template = BasisTemplate::from_raw(variable_basis);
        let coeff_tensor = Coefficients::new(&output_polynomials, &basis_template, &device);

        Self {
            coeff_tensor,
            basis_template,
            device,
        }
    }

    pub fn predict(&self, inputs: &[f32]) -> Vec<f32> {
        let basis = self.basis_template.make_tensor::<B>(
            inputs.iter().enumerate().map(|(p, v)| (p, *v)),
            &self.device,
        );

        let result = self.coeff_tensor.inner().clone().matmul(basis);

        // Flatten and convert to Vec<f32>
        let shape = result.shape();
        let flattened = result.reshape([shape.dims[0] * shape.dims[1]]);
        let data = flattened.to_data();
        data.as_slice::<f32>().unwrap().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    type TestBackend = NdArray;

    #[test]
    fn burn_scratch() {
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

        let device = burn::backend::ndarray::NdArrayDevice::default();
        let burn_net = BurnNetwork::<TestBackend>::from_topology(&topology, device);

        let res = burn_net.predict(&[3.0, 2.0]);
        println!("burn_net result: {:?}", res);
    }

    #[test]
    fn burn_scratch_two() {
        let topology = PolyNetworkTopology::new(2, 2, MutationChances::none(), &mut rand::rng());

        println!("here 1");
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let burn_net = BurnNetwork::<TestBackend>::from_topology(&topology, device);

        let res = burn_net.predict(&[3.0, 2.0]);
        println!("burn_net result: {:?}", res);
    }
}
