use candle_neat::prelude::*;
use tracing::info;
fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    tracing::info!("test");
    let mut running_topology = NetworkTopology::new(2, 2, 3, &mut rand::thread_rng());

    #[allow(unused_assignments)]
    let mut running_network = running_topology.to_network();

    for gen in 0..10000 {
        info!("===NEW GEN ({}) ===", gen);
        running_topology = running_topology.replicate(&mut rand::thread_rng());

        //let debug_info = format!("{:#?}", running_topology);

        //fs::write(format!("./outputs/org_{}.dbg", gen), debug_info).unwrap();

        running_network = running_topology.to_network();
        let _ = running_network.predict(&[1., 5.]);

        info!(
            "===END GEN ({}) === network_len: ({}, {}, {})\n",
            gen,
            running_network.num_nodes(),
            running_network.num_inputs(),
            running_network.num_outputs()
        );
    }

    println!("final topology:\n{:#?}", running_topology)
}
