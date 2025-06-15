use burn::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;
use std::hash::Hash;

use crate::poly::burn_net::expander::{PolyComponent, Polynomial, Variable};

#[cfg(test)]
mod tests;

/// GPU-based polynomial representation using tensors
///
/// Represents a polynomial as:
/// - coefficients: Tensor of shape [max_terms] containing the coefficient of each term
/// - exponents: Tensor of shape [max_terms, max_vars] containing exponents for each variable in each term
/// - valid_mask: Tensor of shape [max_terms] indicating which terms are valid (not padding)
#[derive(Debug, Clone)]
pub struct GpuPolynomial<B: Backend> {
    device: Device<B>,
    coefficients: Tensor<B, 1>,
    exponents: Tensor<B, 2, Int>,
    valid_mask: Tensor<B, 1, Bool>,
    var_count: usize,
    max_terms: usize,
}

impl<B: Backend> GpuPolynomial<B> {
    /// Create a new empty polynomial
    pub fn new(device: &Device<B>, max_terms: usize, var_count: usize) -> Self {
        let coefficients = Tensor::zeros([max_terms], device);
        let exponents = Tensor::zeros([max_terms, var_count], device);
        let valid_mask = Tensor::<B, 1>::zeros([max_terms], device).greater_elem(0.5);

        Self {
            device: device.clone(),
            coefficients,
            exponents,
            valid_mask,
            var_count,
            max_terms,
        }
    }

    /// Create a polynomial from components (for compatibility with CPU version)
    pub fn from_components(
        device: &Device<B>,
        components: Vec<(f32, Vec<(usize, i32)>)>,
        var_count: usize,
    ) -> Self {
        let max_terms = components.len().max(32); // Minimum size for GPU efficiency
        let mut coeffs = vec![0.0f32; max_terms];
        let mut exps = vec![0i32; max_terms * var_count];
        let mut valid = vec![0.0f32; max_terms];

        for (i, (coeff, vars)) in components.into_iter().enumerate() {
            coeffs[i] = coeff;
            valid[i] = 1.0;

            for (var_idx, exp) in vars {
                if var_idx < var_count {
                    exps[i * var_count + var_idx] = exp;
                }
            }
        }

        let coefficients = Tensor::<B, 1>::from_data(coeffs.as_slice(), device);
        let exponents =
            Tensor::<B, 1, Int>::from_data(exps.as_slice(), device).reshape([max_terms, var_count]);
        let valid_mask = Tensor::<B, 1>::from_data(valid.as_slice(), device).greater_elem(0.5);

        Self {
            device: device.clone(),
            coefficients,
            exponents,
            valid_mask,
            var_count,
            max_terms,
        }
    }

    /// Create a simple monomial (coefficient * variable^exponent)
    pub fn monomial(
        device: &Device<B>,
        coeff: f32,
        var_idx: usize,
        exponent: i32,
        var_count: usize,
    ) -> Self {
        let max_terms = 32;
        let mut coeffs = vec![0.0f32; max_terms];
        let mut exps = vec![0i32; max_terms * var_count];
        let mut valid = vec![0.0f32; max_terms];

        coeffs[0] = coeff;
        valid[0] = 1.0;
        if var_idx < var_count {
            exps[var_idx] = exponent;
        }

        let coefficients = Tensor::<B, 1>::from_data(coeffs.as_slice(), device);
        let exponents =
            Tensor::<B, 1, Int>::from_data(exps.as_slice(), device).reshape([max_terms, var_count]);
        let valid_mask = Tensor::<B, 1>::from_data(valid.as_slice(), device).greater_elem(0.5);

        Self {
            device: device.clone(),
            coefficients,
            exponents,
            valid_mask,
            var_count,
            max_terms,
        }
    }

    /// Add two polynomials
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.var_count, other.var_count, "Variable count mismatch");

        // Concatenate terms from both polynomials
        let new_max_terms = (self.max_terms + other.max_terms).max(32);

        // Pad if necessary
        let self_coeffs = pad_tensor(&self.coefficients, new_max_terms, 0.0);
        let other_coeffs = pad_tensor(&other.coefficients, new_max_terms, 0.0);

        let self_exps = pad_tensor_2d(&self.exponents, new_max_terms, self.var_count, 0);
        let other_exps = pad_tensor_2d(&other.exponents, new_max_terms, self.var_count, 0);

        let self_valid = pad_tensor_bool(&self.valid_mask, new_max_terms, false);
        let other_valid = pad_tensor_bool(&other.valid_mask, new_max_terms, false);

        // Combine terms
        let coefficients =
            Tensor::cat(vec![self_coeffs, other_coeffs], 0).slice([0..new_max_terms]);
        let exponents = Tensor::cat(vec![self_exps, other_exps], 0)
            .slice([0..new_max_terms, 0..self.var_count]);
        let valid_mask = Tensor::cat(vec![self_valid, other_valid], 0).slice([0..new_max_terms]);

        let mut result = Self {
            device: self.device.clone(),
            coefficients,
            exponents,
            valid_mask,
            var_count: self.var_count,
            max_terms: new_max_terms,
        };

        result.simplify();
        result
    }

    /// Multiply two polynomials (FOIL expansion)
    pub fn multiply(&self, other: &Self) -> Self {
        assert_eq!(self.var_count, other.var_count, "Variable count mismatch");

        let new_max_terms = (self.max_terms * other.max_terms).max(32);
        let mut result = Self::new(&self.device, new_max_terms, self.var_count);

        // Get valid indices for both polynomials
        let self_valid_indices = get_valid_indices(&self.valid_mask);
        let other_valid_indices = get_valid_indices(&other.valid_mask);

        // Check if any valid indices exist
        let self_count = self_valid_indices.dims()[0];
        let other_count = other_valid_indices.dims()[0];

        if self_count == 0 || other_count == 0 {
            return result;
        }

        // Perform multiplication on GPU using broadcasting
        // Extract valid terms
        let self_coeffs = self
            .coefficients
            .clone()
            .gather(0, self_valid_indices.clone());
        let self_exps = self
            .exponents
            .clone()
            .slice([0..self_count, 0..self.var_count]);

        let other_coeffs = other
            .coefficients
            .clone()
            .gather(0, other_valid_indices.clone());
        let other_exps = other
            .exponents
            .clone()
            .slice([0..other_count, 0..self.var_count]);

        // Compute all products using broadcasting
        let n_self = self_count;
        let n_other = other_count;

        // Reshape for broadcasting
        let self_coeffs_expanded = self_coeffs.clone().reshape([n_self, 1]);
        let other_coeffs_expanded = other_coeffs.clone().reshape([1, n_other]);

        // Multiply coefficients
        let prod_coeffs = self_coeffs_expanded
            .mul(other_coeffs_expanded)
            .flatten(0, 1);

        // Add exponents
        let self_exps_expanded = self_exps.clone().reshape([n_self, 1, self.var_count]);
        let other_exps_expanded = other_exps.clone().reshape([1, n_other, self.var_count]);

        let prod_exps = self_exps_expanded
            .add(other_exps_expanded)
            .reshape([n_self * n_other, self.var_count]);

        // Create result polynomial
        let n_products = n_self * n_other;
        if n_products <= new_max_terms {
            result.coefficients = pad_tensor(&prod_coeffs, new_max_terms, 0.0);
            result.exponents = pad_tensor_2d(&prod_exps, new_max_terms, self.var_count, 0);

            let valid = Tensor::<B, 1>::ones([n_products], &self.device);
            result.valid_mask = pad_tensor_bool(&valid.greater_elem(0.5), new_max_terms, false);
        }

        result.simplify();
        result
    }

    /// Raise polynomial to a power
    pub fn pow(&self, exponent: i32) -> Self {
        if exponent == 0 {
            // Return constant 1
            let mut result = Self::new(&self.device, 32, self.var_count);
            let ones = vec![1.0f32];
            let coeffs = Tensor::<B, 1>::from_data(ones.as_slice(), &self.device);
            result.coefficients = pad_tensor(&coeffs, 32, 0.0);
            let valid = Tensor::<B, 1>::ones([1], &self.device).greater_elem(0.5);
            result.valid_mask = pad_tensor_bool(&valid, 32, false);
            return result;
        }

        if exponent == 1 {
            return self.clone();
        }

        let mut result = self.clone();
        let abs_exp = exponent.abs();

        for _ in 1..abs_exp {
            result = result.multiply(self);
        }

        if exponent < 0 {
            result.invert();
        }

        result
    }

    /// Invert polynomial (multiply all exponents by -1)
    fn invert(&mut self) {
        self.exponents = self.exponents.clone().mul_scalar(-1);
    }

    /// Simplify by combining like terms
    pub fn simplify(&mut self) {
        // This is challenging to do efficiently on GPU due to the need for sorting and grouping
        // For now, we'll do a simple GPU-based approach that may not be fully optimal

        // Get valid indices
        let valid_indices = get_valid_indices(&self.valid_mask);
        if valid_indices.dims()[0] <= 1 {
            return;
        }

        // For true GPU efficiency, we'd need a parallel sorting algorithm
        // For this implementation, we'll focus on demonstrating the concept

        // Compact the polynomial by removing zero-coefficient terms
        let nonzero_mask = self.coefficients.clone().abs().greater_elem(1e-10);
        let combined_mask = self
            .valid_mask
            .clone()
            .int()
            .mul(nonzero_mask.int())
            .greater_elem(0);
        self.valid_mask = combined_mask;
    }

    /// Convert back to CPU representation for compatibility
    pub fn to_components(&self) -> Vec<(f32, Vec<(usize, i32)>)> {
        let coeffs = self.coefficients.clone().to_data();
        let exps = self.exponents.clone().to_data();
        let valid = self.valid_mask.clone().to_data();

        let mut components = Vec::new();

        let coeffs_vec = coeffs.to_vec::<f32>().unwrap();
        let exps_data = exps.to_vec::<i32>().unwrap();
        //let exps_data = exps.to_vec::<i32>().unwrap();
        let valid_vec = valid.to_vec::<u32>().unwrap();

        for i in 0..self.max_terms {
            if valid_vec[i] > 0 && coeffs_vec[i].abs() > 1e-10 {
                let mut vars = Vec::new();
                for j in 0..self.var_count {
                    let exp = exps_data[i * self.var_count + j] as i32;
                    if exp != 0 {
                        vars.push((j, exp));
                    }
                }
                components.push((coeffs_vec[i], vars));
            }
        }

        components
    }
}

// Helper functions

fn pad_tensor<B: Backend>(tensor: &Tensor<B, 1>, new_size: usize, pad_value: f32) -> Tensor<B, 1> {
    let current_size = tensor.dims()[0];
    if current_size >= new_size {
        return tensor.clone().slice([0..new_size]);
    }

    let padding = new_size - current_size;
    let pad_data = vec![pad_value; padding];
    let pad_tensor = Tensor::<B, 1>::from_data(pad_data.as_slice(), &tensor.device());
    Tensor::cat(vec![tensor.clone(), pad_tensor], 0)
}

fn pad_tensor_2d<B: Backend>(
    tensor: &Tensor<B, 2, Int>,
    new_rows: usize,
    cols: usize,
    pad_value: i32,
) -> Tensor<B, 2, Int> {
    let current_rows = tensor.dims()[0];
    if current_rows >= new_rows {
        return tensor.clone().slice([0..new_rows, 0..cols]);
    }

    // For 2D padding, we need to pad row by row
    let padding = new_rows - current_rows;
    let zeros = Tensor::<B, 2, Int>::zeros([padding, cols], &tensor.device());
    Tensor::cat(vec![tensor.clone(), zeros], 0)
}

fn pad_tensor_bool<B: Backend>(
    tensor: &Tensor<B, 1, Bool>,
    new_size: usize,
    pad_value: bool,
) -> Tensor<B, 1, Bool> {
    let current_size = tensor.dims()[0];
    if current_size >= new_size {
        return tensor.clone().slice([0..new_size]);
    }

    let padding = new_size - current_size;
    let pad_val = if pad_value { 1.0 } else { 0.0 };
    let pad_data = vec![pad_val; padding];
    let pad_tensor =
        Tensor::<B, 1>::from_data(pad_data.as_slice(), &tensor.device()).greater_elem(0.5);
    Tensor::cat(vec![tensor.clone(), pad_tensor], 0)
}

fn get_valid_indices<B: Backend>(valid_mask: &Tensor<B, 1, Bool>) -> Tensor<B, 1, Int> {
    // Get indices where valid_mask is true
    // Convert bool to int (1 for true, 0 for false)
    let mask_int = valid_mask.clone().int();

    // Create indices tensor
    let indices = Tensor::arange(0..valid_mask.dims()[0] as i64, &valid_mask.device());

    // Multiply indices by mask to zero out invalid indices
    let masked_indices = indices.clone().mul(mask_int.clone());

    // Get non-zero indices
    // Note: This is a simplified version. A full implementation would need proper filtering
    masked_indices
}

/// Convert from CPU Polynomial to GPU representation
pub fn from_cpu_polynomial<B: Backend>(
    poly: Polynomial<usize>,
    device: &Device<B>,
    var_count: usize,
) -> GpuPolynomial<B> {
    let components: Vec<(f32, Vec<(usize, i32)>)> = poly
        .into_components()
        .into_iter()
        .map(|comp| {
            let vars = comp
                .operands()
                .iter()
                .map(|var| (*var.var(), var.exponent()))
                .collect();
            (comp.weight(), vars)
        })
        .collect();

    GpuPolynomial::from_components(device, components, var_count)
}

/// Convert from GPU to CPU Polynomial representation
pub fn to_cpu_polynomial<B: Backend>(gpu_poly: GpuPolynomial<B>) -> Polynomial<usize> {
    let mut poly = Polynomial::new();

    for (coeff, vars) in gpu_poly.to_components() {
        let mut component = PolyComponent::base(coeff);
        for (var_idx, exp) in vars {
            component = component.with_operand(var_idx, exp);
        }
        poly.handle_polycomponent(component);
    }

    poly
}
