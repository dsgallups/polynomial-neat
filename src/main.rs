use candle_neat::{prelude::*, topology::mutation::MutationChances};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tracing::info;
fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    tracing::info!("test");
    let mutation_chances = MutationChances::new_from_raw(3, 80., 50., 5., 60., 20.);

    let mut running_topology =
        NetworkTopology::new(2, 2, mutation_chances, &mut rand::thread_rng());

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

    loop {
        info!("looping final network");

        let result = running_network.predict(&[1., 5.]).collect::<Vec<f32>>();
        info!(
            "\nresult: {:?}, network_len: ({}, {}, {})\n===END GEN ({}) ===",
            result,
            running_network.num_nodes(),
            running_network.num_inputs(),
            running_network.num_outputs(),
            gen,
        );
    }
}

#[test]
fn test_something() {
    let res = [].par_iter().sum::<f32>();
    println!("res: {}", res)
}
