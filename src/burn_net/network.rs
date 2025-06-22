use super::{basis_prime::BasisTemplate, coeff::Coefficients, get_topology_polynomials};
use crate::{
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
use tracing::info;
use uuid::Uuid;

/// GPU-accelerated polynomial neural network using the Burn deep learning framework.
///
/// This struct provides high-performance network inference on GPU devices (CUDA, WGPU)
/// or CPU devices. It represents a polynomial network as tensor operations for efficient
/// parallel computation.
///
/// # Type Parameters
///
/// * `B` - The Burn backend to use (e.g., `Cuda`, `Wgpu`, `NdArray`)
///
/// # Architecture
///
/// The network is represented using:
/// - **Coefficient Tensor**: A 2D tensor storing all network weights and biases
/// - **Basis Template**: The structure defining how inputs are transformed through polynomials
/// - **Device**: The compute device (CPU/GPU) where tensors are allocated
///
/// # Example
///
/// ```rust
/// use polynomial_neat::prelude::*;
/// use polynomial_neat::burn_net::network::BurnNetwork;
/// use polynomial_neat::topology::mutation::MutationChances;
/// use burn::backend::NdArray;
///
/// // Create a topology
/// let mutations = MutationChances::new(50);
/// let topology = PolyNetworkTopology::new(3, 2, mutations, &mut rand::rng());
///
/// // Create network using CPU backend for testing
/// let device = burn::backend::ndarray::NdArrayDevice::default();
/// let network = BurnNetwork::<NdArray>::from_topology(&topology, device);
///
/// // Run inference
/// let outputs = network.predict(&[1.0, 2.0, 3.0]);
/// assert_eq!(outputs.len(), 2); // Two output neurons
/// ```
pub struct BurnNetwork<B: Backend> {
    coeff_tensor: Coefficients<B>,
    basis_template: BasisTemplate<usize>,
    device: B::Device,
}

impl<B: Backend> BurnNetwork<B> {
    /// Create a GPU-accelerated network from a topology representation.
    ///
    /// This method converts a `PolyNetworkTopology` into an efficient tensor-based
    /// representation suitable for GPU computation.
    ///
    /// # Arguments
    ///
    /// * `topology` - The network topology to convert
    /// * `device` - The compute device to use (e.g., CUDA, WGPU, CPU)
    ///
    /// # Returns
    ///
    /// A new `BurnNetwork` ready for inference on the specified device
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polynomial_neat::prelude::*;
    /// # use polynomial_neat::burn_net::network::BurnNetwork;
    /// # use polynomial_neat::topology::mutation::MutationChances;
    /// use burn::backend::NdArray;
    ///
    /// let mutations = MutationChances::new(50);
    /// let topology = PolyNetworkTopology::new(4, 3, mutations, &mut rand::rng());
    ///
    /// // Create network on CPU backend
    /// let device = burn::backend::ndarray::NdArrayDevice::default();
    /// let network = BurnNetwork::<NdArray>::from_topology(&topology, device);
    /// ```
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

        info!("Polynomial List:");
        for poly in output_polynomials.iter() {
            info!("{}", poly);
        }

        let variable_basis = basis_from_poly_list(&output_polynomials);

        let basis_template = BasisTemplate::from_raw(variable_basis);

        info!("Basis:\n{basis_template}");
        let coeff_tensor = Coefficients::new(&output_polynomials, &basis_template, &device);

        Self {
            coeff_tensor,
            basis_template,
            device,
        }
    }

    /// Perform a forward pass through the network with the given inputs.
    ///
    /// This method executes the polynomial computations on the specified device
    /// (CPU/GPU) for maximum performance.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Slice of input values. Length should match the number of input neurons.
    ///
    /// # Returns
    ///
    /// A vector containing the output values from the network's output neurons.
    ///
    /// # Panics
    ///
    /// Panics if the number of inputs doesn't match the network's expected input size.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polynomial_neat::prelude::*;
    /// # use polynomial_neat::burn_net::network::BurnNetwork;
    /// # use polynomial_neat::topology::mutation::MutationChances;
    /// # use burn::backend::NdArray;
    /// # let mutations = MutationChances::new(50);
    /// # let topology = PolyNetworkTopology::new(2, 1, mutations, &mut rand::rng());
    /// # let device = burn::backend::ndarray::NdArrayDevice::default();
    /// # let network = BurnNetwork::<NdArray>::from_topology(&topology, device);
    /// // Predict with two inputs
    /// let outputs = network.predict(&[1.0, 0.5]);
    /// assert_eq!(outputs.len(), 1); // One output neuron
    /// ```
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
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(3819234);
        let topology = PolyNetworkTopology::new(2, 2, MutationChances::none(), &mut rng);

        println!("here 1");
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let burn_net = BurnNetwork::<TestBackend>::from_topology(&topology, device);

        let res = burn_net.predict(&[3.0, 2.0]);
        println!("burn_net result: {:?}", res);
    }
}
