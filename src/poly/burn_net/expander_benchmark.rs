use burn::backend::{NdArray, Wgpu};
use burn::prelude::*;
use std::time::Instant;

use crate::poly::burn_net::expander::{PolyComponent, Polynomial, Variable};
use crate::poly::burn_net::gpu_expander::{GpuPolynomial, from_cpu_polynomial, to_cpu_polynomial};

/// Benchmark results for a single operation
#[derive(Debug, Clone)]
struct BenchmarkResult {
    operation: String,
    cpu_time_ms: f64,
    gpu_time_ms: f64,
    speedup: f64,
}

impl BenchmarkResult {
    fn new(operation: String, cpu_time_ms: f64, gpu_time_ms: f64) -> Self {
        let speedup = cpu_time_ms / gpu_time_ms;
        Self {
            operation,
            cpu_time_ms,
            gpu_time_ms,
            speedup,
        }
    }

    fn print(&self) {
        println!(
            "{:<30} | CPU: {:>8.3}ms | GPU: {:>8.3}ms | Speedup: {:>6.2}x {}",
            self.operation,
            self.cpu_time_ms,
            self.gpu_time_ms,
            self.speedup,
            if self.speedup > 1.0 { "↑" } else { "↓" }
        );
    }
}

/// Run a benchmark on both CPU and GPU
fn benchmark_operation<B: Backend, F, G>(
    name: &str,
    cpu_op: F,
    gpu_op: G,
    iterations: u32,
) -> BenchmarkResult
where
    F: Fn(),
    G: Fn(),
{
    // Warmup
    for _ in 0..10 {
        cpu_op();
        gpu_op();
    }

    // CPU benchmark
    let cpu_start = Instant::now();
    for _ in 0..iterations {
        cpu_op();
    }
    let cpu_elapsed = cpu_start.elapsed();
    let cpu_time_ms = cpu_elapsed.as_secs_f64() * 1000.0 / iterations as f64;

    // GPU benchmark
    let gpu_start = Instant::now();
    for _ in 0..iterations {
        gpu_op();
    }
    let gpu_elapsed = gpu_start.elapsed();
    let gpu_time_ms = gpu_elapsed.as_secs_f64() * 1000.0 / iterations as f64;

    BenchmarkResult::new(name.to_string(), cpu_time_ms, gpu_time_ms)
}

/// Generate a random polynomial with given parameters
fn generate_polynomial(num_terms: usize, num_vars: usize, max_degree: i32) -> Polynomial<usize> {
    use rand::Rng;
    let mut rng = rand::rng();
    let mut poly = Polynomial::new();

    for _ in 0..num_terms {
        let coeff = rng.random_range(-10.0..10.0);
        let mut component = PolyComponent::base(coeff);

        // Add random variables with random exponents
        let vars_in_term = rng.random_range(0..=num_vars.min(3));
        for _ in 0..vars_in_term {
            let var = rng.random_range(0..num_vars);
            let exp = rng.random_range(0..=max_degree);
            if exp > 0 {
                component = component.with_operand(var, exp);
            }
        }

        poly.handle_polycomponent(component);
    }

    poly
}

pub fn run_benchmarks<B: Backend>(device: &Device<B>) {
    println!("\n=== Polynomial Expansion Benchmarks: CPU vs GPU ===\n");
    println!("Backend: {:?}", B::name(device));
    println!("Device: {:?}\n", device);

    let mut results = Vec::new();

    // Test 1: Simple polynomial addition
    {
        let poly1 = generate_polynomial(10, 3, 3);
        let poly2 = generate_polynomial(10, 3, 3);

        let gpu_poly1 = from_cpu_polynomial::<B>(poly1.clone(), device, 3);
        let gpu_poly2 = from_cpu_polynomial::<B>(poly2.clone(), device, 3);

        let result = benchmark_operation::<B, _, _>(
            "Simple Addition (10 terms)",
            || {
                let mut p = poly1.clone();
                for comp in poly2.components() {
                    p.handle_polycomponent(comp.clone());
                }
            },
            || {
                let _ = gpu_poly1.add(&gpu_poly2);
            },
            100,
        );
        results.push(result);
    }

    // Test 2: Large polynomial addition
    {
        let poly1 = generate_polynomial(100, 5, 5);
        let poly2 = generate_polynomial(100, 5, 5);

        let gpu_poly1 = from_cpu_polynomial::<B>(poly1.clone(), device, 5);
        let gpu_poly2 = from_cpu_polynomial::<B>(poly2.clone(), device, 5);

        let result = benchmark_operation::<B, _, _>(
            "Large Addition (100 terms)",
            || {
                let mut p = poly1.clone();
                for comp in poly2.components() {
                    p.handle_polycomponent(comp.clone());
                }
            },
            || {
                let _ = gpu_poly1.add(&gpu_poly2);
            },
            100,
        );
        results.push(result);
    }

    // Test 3: Polynomial multiplication (FOIL)
    {
        let poly1 = generate_polynomial(5, 2, 2);
        let poly2 = generate_polynomial(5, 2, 2);

        let gpu_poly1 = from_cpu_polynomial::<B>(poly1.clone(), device, 2);
        let gpu_poly2 = from_cpu_polynomial::<B>(poly2.clone(), device, 2);

        let result = benchmark_operation::<B, _, _>(
            "Multiplication (5x5 terms)",
            || {
                let _ = poly1.clone().mul_expand(&poly2);
            },
            || {
                let _ = gpu_poly1.multiply(&gpu_poly2);
            },
            50,
        );
        results.push(result);
    }

    // Test 4: Large polynomial multiplication
    {
        let poly1 = generate_polynomial(20, 3, 3);
        let poly2 = generate_polynomial(20, 3, 3);

        let gpu_poly1 = from_cpu_polynomial::<B>(poly1.clone(), device, 3);
        let gpu_poly2 = from_cpu_polynomial::<B>(poly2.clone(), device, 3);

        let result = benchmark_operation::<B, _, _>(
            "Large Multiplication (20x20)",
            || {
                let _ = poly1.clone().mul_expand(&poly2);
            },
            || {
                let _ = gpu_poly1.multiply(&gpu_poly2);
            },
            20,
        );
        results.push(result);
    }

    // Test 5: Polynomial exponentiation
    {
        let poly = generate_polynomial(3, 2, 2);
        let gpu_poly = from_cpu_polynomial::<B>(poly.clone(), device, 2);

        let result = benchmark_operation::<B, _, _>(
            "Exponentiation (3 terms)^3",
            || {
                let mut p = Polynomial::new();
                p.expand(poly.clone(), 1.0, 3);
            },
            || {
                let _ = gpu_poly.pow(3);
            },
            50,
        );
        results.push(result);
    }

    // Test 6: Complex polynomial operations
    {
        let poly1 = generate_polynomial(10, 4, 3);
        let poly2 = generate_polynomial(8, 4, 2);
        let poly3 = generate_polynomial(5, 4, 4);

        let gpu_poly1 = from_cpu_polynomial::<B>(poly1.clone(), device, 4);
        let gpu_poly2 = from_cpu_polynomial::<B>(poly2.clone(), device, 4);
        let gpu_poly3 = from_cpu_polynomial::<B>(poly3.clone(), device, 4);

        let result = benchmark_operation::<B, _, _>(
            "Complex: (P1*P2)+P3",
            || {
                let prod = poly1.clone().mul_expand(&poly2);
                let mut result = prod;
                for comp in poly3.components() {
                    result.handle_polycomponent(comp.clone());
                }
            },
            || {
                let prod = gpu_poly1.multiply(&gpu_poly2);
                let _ = prod.add(&gpu_poly3);
            },
            20,
        );
        results.push(result);
    }

    // Test 7: Conversion overhead
    {
        let poly = generate_polynomial(50, 5, 4);

        let result = benchmark_operation::<B, _, _>(
            "CPU↔GPU Conversion (50 terms)",
            || {
                // Just clone as baseline
                let _ = poly.clone();
            },
            || {
                let gpu_poly = from_cpu_polynomial::<B>(poly.clone(), device, 5);
                let _ = to_cpu_polynomial(gpu_poly);
            },
            100,
        );
        results.push(result);
    }

    // Print results
    println!(
        "{:<30} | {:<12} | {:<12} | {:<10}",
        "Operation", "CPU Time", "GPU Time", "Speedup"
    );
    println!("{}", "-".repeat(75));

    for result in &results {
        result.print();
    }

    // Summary statistics
    let avg_speedup = results.iter().map(|r| r.speedup).sum::<f64>() / results.len() as f64;
    let ops_faster_on_gpu = results.iter().filter(|r| r.speedup > 1.0).count();

    println!("\n{}", "=".repeat(75));
    println!("Summary:");
    println!("  Average speedup: {:.2}x", avg_speedup);
    println!(
        "  Operations faster on GPU: {}/{}",
        ops_faster_on_gpu,
        results.len()
    );

    if avg_speedup < 1.0 {
        println!("\n⚠️  Overall, CPU implementation is faster for these polynomial operations.");
        println!("   This is expected due to:");
        println!("   - Small problem sizes");
        println!("   - Irregular memory access patterns");
        println!("   - GPU memory transfer overhead");
        println!("   - Sequential nature of symbolic algebra");
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::Wgpu;

    use super::*;

    #[test]
    #[ignore] // Run with: cargo test --release -- --ignored --nocapture
    fn benchmark_cpu_vs_gpu_ndarray() {
        type B = NdArray;
        let device = Default::default();
        run_benchmarks::<B>(&device);
    }

    #[test]
    #[ignore]
    fn benchmark_cpu_vs_gpu_wgpu() {
        type B = Wgpu;
        let device = Default::default();
        run_benchmarks::<B>(&device);
    }
}

/// Main entry point for running benchmarks
pub fn main() {
    // Run with default backend
    type DefaultBackend = NdArray;
    let device = Default::default();

    println!("Running polynomial expansion benchmarks...");
    run_benchmarks::<DefaultBackend>(&device);

    {
        println!("\n\nRunning with WGPU backend...");
        run_benchmarks::<Wgpu>(&Default::default());
    }
}
