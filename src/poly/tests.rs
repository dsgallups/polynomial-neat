use std::collections::HashSet;

use crate::poly::{candle_net::network::CandleNetwork, prelude::*};
fn _test_dupes() {
    let mutation_chances = MutationChances::new_from_raw(3, 80., 50., 5., 60., 20.);
    let mut top_1 = PolyNetworkTopology::new(20, 20, mutation_chances, &mut rand::thread_rng());

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

#[test]
fn test_two() {
    use crate::{poly::prelude::*, poly::topology::mutation::MutationChances};
    let mutation_chances = MutationChances::new_from_raw(3, 80., 50., 5., 60., 20.);

    let mut running_topology =
        PolyNetworkTopology::new(2, 2, mutation_chances, &mut rand::thread_rng());

    let mut gen = 0;
    println!("here");
    loop {
        println!("===NEW GEN ({}) ===", gen);
        running_topology = running_topology.replicate(&mut rand::thread_rng());

        //let debug_info = format!("{:#?}", running_topology);

        //fs::write(format!("./outputs/org_{}.dbg", gen), debug_info).unwrap();

        let running_network = running_topology.to_simple_network();
        let candle_network =
            CandleNetwork::from_topology(&running_topology, &candle_core::Device::Cpu).unwrap();
        println!("simple network made");
        let result = running_network.predict(&[1., 5.]).collect::<Vec<f32>>();
        let candle_result = candle_network
            .predict(&[1., 5.])
            .unwrap()
            .collect::<Vec<_>>();

        println!(
            "\nresult: {:?},candle_result: {:?} network_len: ({}, {}, {})\n===END GEN ({}) ===",
            result,
            candle_result,
            running_network.num_nodes(),
            running_network.num_inputs(),
            running_network.num_outputs(),
            gen,
        );

        gen += 1;
        /*if gen > 1000 {
            break;
        }*/
    }
}

fn _test_inf() {
    use crate::{poly::prelude::*, poly::topology::mutation::MutationChances};
    use tracing::info;
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    tracing::info!("test");
    let mutation_chances = MutationChances::new_from_raw(3, 80., 50., 5., 60., 20.);

    let mut running_topology =
        PolyNetworkTopology::new(2, 2, mutation_chances, &mut rand::thread_rng());

    #[allow(unused_assignments)]
    let mut running_network = running_topology.to_simple_network();

    let mut gen = 0;
    loop {
        info!("===NEW GEN ({}) ===", gen);
        running_topology = running_topology.replicate(&mut rand::thread_rng());

        //let debug_info = format!("{:#?}", running_topology);

        //fs::write(format!("./outputs/org_{}.dbg", gen), debug_info).unwrap();

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
            gen,
        );
        gen += 1;
        /*if gen > 1000 {
            break;
        }*/
    }
}

#[test]
fn test_something() {
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
    let res = [].par_iter().sum::<f32>();
    println!("res: {}", res)
}
