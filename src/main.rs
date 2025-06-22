use std::io::{self, Write};

use burn::backend::Cuda;
// use burn::backend::{Cuda, Wgpu};
use polynomial_neat::{
    burn_net::network::BurnNetwork, prelude::*, topology::mutation::MutationChances,
};
use rand::{SeedableRng, rngs::StdRng};
use tracing::info;

//const MAX_GEN: i32 = 10;
const MAX_GEN: i32 = 5000;

fn main() {
    tracing_subscriber::fmt::init();
    let mutation_chances = MutationChances::new_from_raw(5, 80., 50., 5., 60., 0.);

    let mut seed = 0;
    let mut rng = StdRng::seed_from_u64(0);

    let mut running_topology = PolyNetworkTopology::new(2, 2, mutation_chances, &mut rng);

    //let device = burn::backend::wgpu::WgpuDevice::DiscreteGpu(0);
    let device = burn::backend::cuda::CudaDevice::default();

    let mut generation = 0;
    let mut step_for_seed = true;
    let mut step = false;
    loop {
        let mut next = running_topology.replicate(&mut rng);

        loop {
            let info = next.info();

            if info.num_inputs == 0 || info.num_outputs == 0 {
                next = running_topology.replicate(&mut rng);
                continue;
            }
            break;
        }

        let info = next.info();
        info!("Topology passed! {:?}", info);

        running_topology = next;

        let running_network = running_topology.to_simple_network();

        let result = running_network.predict(&[1., 5.]).collect::<Vec<f32>>();
        let burn_network = BurnNetwork::<Cuda>::from_topology(&running_topology, device.clone());
        let burn_result = burn_network.predict(&[1., 5.]);

        let mut difference = Vec::new();

        #[allow(clippy::unused_enumerate_index)]
        for (_i, (cpu_p, burn_p)) in result.into_iter().zip(burn_result.into_iter()).enumerate() {
            if (cpu_p - burn_p).abs() > 0.01 {
                difference.push((cpu_p, burn_p));
            }
        }
        if !difference.is_empty() {
            info!("difference found:");
            info!(
                "==network_len==\n\
                num_nodes: {}\n\
                num_inputs: {}\n\
                num_outputs: {}",
                running_network.num_nodes(),
                running_network.num_inputs(),
                running_network.num_outputs()
            );
            let mut output = String::new();
            for (cpu_p, burn_p) in difference {
                output.push_str(&format!("(cpu: {cpu_p}, gpu: {burn_p}), "));
            }
            info!("{}", output);

            info!("Seed: {}", seed);
            step = true;
        }

        if step_for_seed && step {
            info!("Press enter to continue");
            io::stdout().flush().unwrap();
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            step_for_seed = false;
            step = false;
        }

        generation += 1;
        if generation > MAX_GEN {
            running_topology = PolyNetworkTopology::new(2, 20, mutation_chances, &mut rng);
            generation = 0;
            step_for_seed = true;
            step = false;
            seed += 1;
            info!("Generation {seed}");
            rng = StdRng::seed_from_u64(seed);
        }
    }
}
