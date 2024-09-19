use crate::prelude::NetworkTopology;

#[test]
fn test_something() {
    let my_topology = NetworkTopology::new(10, 10, 3, &mut rand::thread_rng());

    let my_network = my_topology.to_network();

    let res = my_network.predict(&[1., 5.]);

    for (i, val) in res.into_iter().enumerate() {
        println!("{}: {}", i, val)
    }
}
