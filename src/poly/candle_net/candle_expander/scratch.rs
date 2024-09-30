use candle_core::{DType, Device, Result, Tensor};
use fnv::FnvHashMap;
use uuid::Uuid;

use crate::poly::{
    candle_net::{basis_prime::BasisTemplate, coeff::Coefficients, get_topology_polynomials},
    prelude::{
        arc, InputTopology, MutationChances, NetworkTopology, NeuronTopology, SimpleNetwork,
    },
};

use super::polynomial::{Indeterminate, PolyComponent, Polynomial};

#[derive(Clone, Copy, PartialOrd, Ord, Debug, PartialEq, Default, Eq)]
struct X;

impl<'dev> Indeterminate<'dev> for X {
    fn apply_operation(self, device: &'dev Device, weight: f32, exponent: i32) -> Polynomial<'dev> {
        let polyc = PolyComponent::simple(weight, 0, exponent);
        Polynomial::from_polycomponent(device, polyc)
    }
    fn identity(self, device: &'dev Device) -> Polynomial<'dev> {
        Polynomial::unit(device, 0)
    }
}

#[test]
fn simple_polynomial_expansion() -> Result<()> {
    let device = Device::Cpu;
    // 4x^3 + 3x^2
    let v = Polynomial::new(&device)
        .add_operation(4., X, 3)
        .add_operation(3., X, 2);

    // (4x^3 + 3x^2)^2
    let h = Polynomial::new(&device).add_operation(1., v, 2);

    //let next =

    Ok(())
}
