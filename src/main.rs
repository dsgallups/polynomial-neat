use candle_neat::{prelude::*, topology::mutation::MutationChances};
use tracing::info;
fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    tracing::info!("test");
    let mutation_chances = MutationChances::new_from_raw(80, 80., 50., 80., 5., 60., 20., 10.);
    let mut running_topology =
        NetworkTopology::new(2, 2, mutation_chances, &mut rand::thread_rng());

    #[allow(unused_assignments)]
    let mut running_network = running_topology.to_network();

    let mut gen = 0;
    loop {
        info!("===NEW GEN ({}) ===", gen);
        running_topology = running_topology.replicate(&mut rand::thread_rng());

        info!("replicated successfully");

        //let debug_info = format!("{:#?}", running_topology);

        //fs::write(format!("./outputs/org_{}.dbg", gen), debug_info).unwrap();

        running_network = running_topology.to_network();

        info!("converted to network");
        let res = running_network.predict(&[1., 5.]).collect::<Vec<_>>();

        info!(
            "\nres: {:?}\n network_len: ({}, {}, {})\n===END GEN ({}) ===\n",
            res,
            running_network.num_nodes(),
            running_network.num_inputs(),
            running_network.num_outputs(),
            gen,
        );
        gen += 1;
    }
}
