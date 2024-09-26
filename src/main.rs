use candle_neat::{
    candle_net::network::CandleNetwork, prelude::*, topology::mutation::MutationChances,
};

fn main() {
    let mutation_chances = MutationChances::new_from_raw(3, 80., 50., 5., 60., 20.);

    let mut running_topology =
        NetworkTopology::new(2, 2, mutation_chances, &mut rand::thread_rng());

    let mut gen = 0;
    println!("here");
    loop {
        println!("===NEW GEN ({}) ===", gen);
        running_topology = running_topology.replicate(&mut rand::thread_rng());

        //let debug_info = format!("{:#?}", running_topology);

        //fs::write(format!("./outputs/org_{}.dbg", gen), debug_info).unwrap();

        let running_network = running_topology.to_simple_network();

        println!("simple network made");
        let result = running_network.predict(&[1., 5.]).collect::<Vec<f32>>();
        println!("simple net predicted");
        let candle_network =
            CandleNetwork::from_topology(&running_topology, &candle_core::Device::Cpu).unwrap();
        println!("candle network made");
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
