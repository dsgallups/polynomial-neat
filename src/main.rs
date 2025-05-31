use std::io::{self, Write};

use candle_core::Device;
use candle_neat::poly::{
    candle_net::network::CandleNetwork, prelude::*, topology::mutation::MutationChances,
};
use tracing::info;

fn main() {
    tracing_subscriber::fmt::init();
    let mutation_chances = MutationChances::new_from_raw(5, 80., 50., 5., 60., 0.);

    let mut running_topology = PolyNetworkTopology::new(2, 2, mutation_chances, &mut rand::rng());

    let dev = Device::new_metal(0).unwrap();
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
        let candle_network = CandleNetwork::from_topology(&running_topology, &dev).unwrap();
        info!("candle network made");
        let candle_result = candle_network
            .predict(&[1., 5.])
            .unwrap()
            .collect::<Vec<_>>();

        info!(
            "==network_len==\n\
            num_nodes: {}\n\
            num_inputs: {}\n\
            num_outputs: {}",
            running_network.num_nodes(),
            running_network.num_inputs(),
            running_network.num_outputs()
        );
        info!("\nsresult: {:?}\ncresult: {:?}", result, candle_result);

        let mut difference = Vec::new();
        for (i, (s_p, c_p)) in result
            .into_iter()
            .zip(candle_result.into_iter())
            .enumerate()
        {
            if (s_p - c_p).abs() > 0.001 {
                difference.push((s_p, c_p));
            }
        }
        if !difference.is_empty() {
            info!("difference found:");
            let mut output = String::new();
            for (s_p, c_p) in difference {
                output.push_str(&format!("({s_p}, {c_p}), "));
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
        if generation > 10 {
            info!("Resetting");
            running_topology = PolyNetworkTopology::new(2, 20, mutation_chances, &mut rand::rng());
            generation = 0;
        }
        /*if generation > 1000 {
            break;
        }*/
    }
}
