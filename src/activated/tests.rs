use std::collections::HashSet;

use crate::activated::prelude::*;
#[test]
fn test_dupes() {
    let mutation_chances = MutationChances::new_from_raw(3, 80., 50., 5., 60., 20., 10.);
    let mut top_1 = NetworkTopology::new(20, 20, mutation_chances, &mut rand::thread_rng());

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
