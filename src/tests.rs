use std::collections::HashSet;

use crate::{burn_net::network::BurnNetwork, prelude::*};
use burn::backend::NdArray;
fn _test_dupes() {
    let mutation_chances = MutationChances::new_from_raw(3, 80., 50., 5., 60., 20.);
    let mut top_1 = PolyNetworkTopology::new(20, 20, mutation_chances, &mut rand::rng());

    let mut top_2 = top_1.deep_clone();

    for _ in 0..100000 {
        let t1_h = top_1.neuron_ids().into_iter().collect::<HashSet<_>>();

        for id in top_2.neuron_ids() {
            assert!(!t1_h.contains(&id))
        }

        top_1 = top_2;
        top_2 = top_1.deep_clone();
    }
}

fn _test_two() {
    use crate::{prelude::*, topology::mutation::MutationChances};
    let mutation_chances = MutationChances::new_from_raw(3, 80., 50., 5., 60., 20.);

    let mut running_topology = PolyNetworkTopology::new(2, 2, mutation_chances, &mut rand::rng());

    let mut generation = 0;
    println!("hk ere");
    loop {
        println!("===NEW GEN ({}) ===", generation);
        running_topology = running_topology.replicate(&mut rand::rng());

        //let debug_info = format!("{:#?}", running_topology);

        //fs::write(format!("./outputs/org_{}.dbg", generation), debug_info).unwrap();

        let running_network = running_topology.to_simple_network();
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let burn_network = BurnNetwork::<NdArray>::from_topology(&running_topology, device);
        println!("simple network made");
        let result = running_network.predict(&[1., 5.]).collect::<Vec<f32>>();
        let burn_result = burn_network.predict(&[1., 5.]);

        println!(
            "\nresult: {:?},burn_result: {:?} network_len: ({}, {}, {})\n===END GEN ({}) ===",
            result,
            burn_result,
            running_network.num_nodes(),
            running_network.num_inputs(),
            running_network.num_outputs(),
            generation,
        );

        generation += 1;
        /*if generation > 1000 {
            break;
        }*/
    }
}

fn _test_inf() {
    use crate::{prelude::*, topology::mutation::MutationChances};
    use tracing::info;
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    tracing::info!("test");
    let mutation_chances = MutationChances::new_from_raw(3, 80., 50., 5., 60., 20.);

    let mut running_topology = PolyNetworkTopology::new(2, 2, mutation_chances, &mut rand::rng());

    #[allow(unused_assignments)]
    let mut running_network = running_topology.to_simple_network();

    let mut generation = 0;
    loop {
        info!("===NEW GEN ({}) ===", generation);
        running_topology = running_topology.replicate(&mut rand::rng());

        //let debug_info = format!("{:#?}", running_topology);

        //fs::write(format!("./outputs/org_{}.dbg", generation), debug_info).unwrap();

        running_network = running_topology.to_simple_network();
        info!("simple network made");
        let result = running_network.predict(&[1., 5.]).collect::<Vec<f32>>();
        //let candle_

        info!(
            "\nresult: {:?}, network_len: ({}, {}, {})\n===END GEN ({}) ===",
            result,
            running_network.num_nodes(),
            running_network.num_inputs(),
            running_network.num_outputs(),
            generation,
        );
        generation += 1;
        if generation > 1000 {
            break;
        }
    }
}

#[test]
fn test_something() {
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
    let res = [].par_iter().sum::<f32>();
    println!("res: {}", res)
}
