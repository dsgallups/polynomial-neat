use candle_core::{DType, Device, Result, Tensor};
use fnv::FnvHashMap;
use uuid::Uuid;

use crate::{
    candle_net::{basis_prime::BasisTemplate, coeff::Coefficients, get_topology_polynomials},
    prelude::{
        arc, InputTopology, MutationChances, NetworkTopology, NeuronTopology, SimpleNetwork,
    },
};

use super::polynomial::{Indeterminate, PolyComponent, Polynomial};

#[derive(Clone, Copy, PartialOrd, Ord, Debug, PartialEq, Default, Eq)]
struct X;

impl<'dev> Indeterminate<'dev> for X {
    type Variable = X;
    fn apply_operation(
        self,
        device: &'dev Device,
        weight: f32,
        exponent: i32,
    ) -> Polynomial<'dev, Self::Variable> {
        let polyc = PolyComponent::simple(weight, self, exponent);
        Polynomial::from_polycomponent(device, polyc)
    }
    fn identity(self, device: &'dev Device) -> Polynomial<'dev, Self::Variable> {
        Polynomial::unit(device, self)
    }
}

#[test]
fn scratch() -> Result<()> {
    let device = Device::Cpu;
    let v = Polynomial::new(&device)
        .with_operation(1., X, 2)
        .with_operation(1., X, 1);

    let h = Polynomial::new(&device).add_operation(1., v, 2);

    //let next =

    Ok(())
}

#[test]
fn old_scratch() -> Result<()> {
    // V(x) = x^2 + x
    /*let v = Polynomial::default()
        .with_operation(1., X, 2)
        .with_operation(1., X, 1);

    // W(x) = 3x + 4
    let w = Polynomial::default()
        .with_operation(3., X, 4)
        .with_polycomponent(PolyComponent::base(4.));

    // L(x) = x^2 + 3x
    let l = Polynomial::default()
        .with_operation(1., X, 2)
        .with_operation(3., X, 1);*/

    //Since the highest power of all these functions is x^2, our basis will be
    //
    let d = Device::Cpu;
    let input_values = vec![2., 5.];

    let inputs = Tensor::new(input_values.as_slice(), &d)?;
    println!("initial inputs: {}", inputs);
    let n = input_values.len();
    let exp = n + 1;
    let exponents = Tensor::arange(0u32, exp as u32, &d)?.to_dtype(DType::F64)?;
    println!("initial exponents: {}\n", exponents);

    let input_col = inputs.reshape((n, 1))?.broadcast_as((n, exp))?;

    let exponents_row = exponents.broadcast_as((n, exp))?;

    println!(
        "input_col: \n{}\nexponents_row:\n{}\n",
        input_col, exponents_row
    );

    /*
        [
            [1, x, x^2],
            [1, y, y^2]
        ]

    */
    let vandermonde = input_col.pow(&exponents_row)?;

    println!("Vandermonde:\n{}\n", vandermonde);
    /*
        [
        [ 1 ].
        [ 0 ],
        [ 1 ]
        ]
        f(x) = x^2 + 1
        []
    */
    let f_x = Tensor::new(&[1., 0., 1.], &d)?.reshape((3, 1))?;
    let f = vandermonde.matmul(&f_x)?;
    /*
        g(x) = 2x^2 + 3x + 2
    */
    let g_x = Tensor::new(&[2., 3., 2.], &d)?.reshape((3, 1))?;
    let g = vandermonde.matmul(&g_x)?;

    //let h = f_x.con
    println!("f = \n{}\n\n g = \n{}\n\n", f, g);

    /*
        [
            []
        ]

        h(x, y) = 3x^2 + 3x + 2y
    */

    Ok(())
}
