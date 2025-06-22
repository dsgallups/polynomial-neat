use std::io::{self, Write};

use burn::backend::Cuda;
// use burn::backend::{Cuda, Wgpu};
use burn_neat::poly::{
    burn_net::network::BurnNetwork, prelude::*, topology::mutation::MutationChances,
};
use rand::{SeedableRng, rngs::StdRng};
use tracing::info;

const MAX_GEN: i32 = 5000;

fn main() {
    tracing_subscriber::fmt::init();
    let mut rng = StdRng::seed_from_u64(12345);
    let mutation_chances = MutationChances::new_from_raw(5, 80., 50., 5., 60., 0.);

    let mut running_topology = PolyNetworkTopology::new(2, 2, mutation_chances, &mut rng);

    //let device = burn::backend::wgpu::WgpuDevice::DiscreteGpu(0);
    let device = burn::backend::cuda::CudaDevice::default();
    let mut generation = 0;
    info!("here");
    let mut step = false;
    loop {
        info!("===NEW GEN ({}) ===", generation);
        running_topology = running_topology.replicate(&mut rand::rng());

        //let debug_info = format!("{:#?}", running_topology);

        //fs::write(format!("./outputs/org_{}.dbg", generation), debug_info).unwrap();

        let running_network = running_topology.to_simple_network();

        info!("simple network made");
        let result = running_network.predict(&[1., 5.]).collect::<Vec<f32>>();
        info!("simple net predicted");
        let burn_network = BurnNetwork::<Cuda>::from_topology(&running_topology, device.clone());
        info!("burn network made");
        let burn_result = burn_network.predict(&[1., 5.]);

        info!(
            "==network_len==\n\
            num_nodes: {}\n\
            num_inputs: {}\n\
            num_outputs: {}",
            running_network.num_nodes(),
            running_network.num_inputs(),
            running_network.num_outputs()
        );
        info!("\ncpu result: {:?}\ngpu result: {:?}", result, burn_result);

        let mut difference = Vec::new();

        #[allow(clippy::unused_enumerate_index)]
        for (_i, (cpu_p, burn_p)) in result.into_iter().zip(burn_result.into_iter()).enumerate() {
            if (cpu_p - burn_p).abs() > 0.01 {
                difference.push((cpu_p, burn_p));
            }
        }
        if !difference.is_empty() {
            info!("difference found:");
            let mut output = String::new();
            for (cpu_p, burn_p) in difference {
                output.push_str(&format!("(cpu: {cpu_p}, gpu: {burn_p}), "));
            }
            info!("{}", output);
            step = true;
        }

        info!("===END GEN ({}) ===", generation);

        if step {
            info!("Press enter to continue");
            io::stdout().flush().unwrap();
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
        }

        generation += 1;
        if generation > MAX_GEN {
            info!("Resetting");
            running_topology = PolyNetworkTopology::new(2, 20, mutation_chances, &mut rand::rng());
            generation = 0;
        }
        /*if generation > 1000 {
            break;
        }*/
    }
}
