use std::{
    collections::HashSet,
    sync::{Arc, RwLock},
};

use rand::Rng;
use uuid::Uuid;

use crate::prelude::*;

#[derive(Clone, Debug)]
/// Represents the topology (structure) of a polynomial neural network.
///
/// This struct encodes the complete architecture of a neural network including:
/// - All neurons (input, hidden, and output)
/// - Connections between neurons
/// - Mutation parameters for evolution
///
/// The topology can be evolved through mutations and converted into executable
/// networks for inference.
///
/// # Example
///
/// ```rust
/// use polynomial_neat::prelude::*;
/// use polynomial_neat::topology::mutation::MutationChances;
///
/// // Create a network with 3 inputs and 2 outputs
/// let mutations = MutationChances::new(50);
/// let topology = PolyNetworkTopology::new(3, 2, mutations, &mut rand::rng());
///
/// // Evolve the network
/// let evolved = topology.replicate(&mut rand::rng());
///
/// // Convert to executable network
/// let network = evolved.to_simple_network();
/// ```
pub struct PolyNetworkTopology {
    neurons: Vec<Arc<RwLock<PolyNeuronTopology>>>,
    mutation_chances: MutationChances,
}

impl PolyNetworkTopology {
    /// Create a network topology from raw components.
    ///
    /// This is a low-level constructor primarily used internally or when
    /// manually constructing network architectures.
    ///
    /// # Arguments
    /// * `neurons` - Vector of all neurons in the network
    /// * `mutation_chances` - Configuration for evolution parameters
    pub fn from_raw_parts(
        neurons: Vec<Arc<RwLock<PolyNeuronTopology>>>,
        mutation_chances: MutationChances,
    ) -> Self {
        Self {
            neurons,
            mutation_chances,
        }
    }

    pub fn new(
        num_inputs: usize,
        num_outputs: usize,
        mutation_chances: MutationChances,
        rng: &mut impl Rng,
    ) -> Self {
        let input_neurons = (0..num_inputs)
            .map(|_| Arc::new(RwLock::new(PolyNeuronTopology::input(Uuid::new_v4()))))
            .collect::<Vec<_>>();

        let output_neurons = (0..num_outputs)
            .map(|_| {
                //a random number of connections to random input neurons;
                let num_connections = if input_neurons.len() == 1 {
                    1 // If only one input, connect to it
                } else {
                    rng.random_range(1..input_neurons.len())
                };
                let mut chosen_inputs = (0..num_connections)
                    .map(|_| {
                        let topology_index = rng.random_range(0..input_neurons.len());
                        let input = input_neurons.get(topology_index).unwrap();
                        (
                            PolyInputTopology::new_rand(Arc::downgrade(input), rng),
                            topology_index,
                        )
                    })
                    .collect::<Vec<_>>();

                chosen_inputs.sort_by_key(|(_, i)| *i);
                chosen_inputs.dedup_by_key(|(_, i)| *i);

                let chosen_inputs = chosen_inputs.into_iter().map(|(input, _)| input).collect();

                Arc::new(RwLock::new(PolyNeuronTopology::output(
                    Uuid::new_v4(),
                    chosen_inputs,
                )))
            })
            .collect::<Vec<_>>();

        let neurons = input_neurons.into_iter().chain(output_neurons).collect();

        Self {
            neurons,
            mutation_chances,
        }
    }

    pub fn new_thoroughly_connected(
        num_inputs: usize,
        num_outputs: usize,
        mutation_chances: MutationChances,
        rng: &mut impl Rng,
    ) -> Self {
        let input_neurons = (0..num_inputs)
            .map(|_| Arc::new(RwLock::new(PolyNeuronTopology::input(Uuid::new_v4()))))
            .collect::<Vec<_>>();

        let output_neurons = (0..num_outputs)
            .map(|_| {
                //every output neuron is connected to every input neuron

                let chosen_inputs = input_neurons
                    .iter()
                    .map(|input| PolyInputTopology::new_rand(Arc::downgrade(input), rng))
                    .collect::<Vec<_>>();

                Arc::new(RwLock::new(PolyNeuronTopology::output(
                    Uuid::new_v4(),
                    chosen_inputs,
                )))
            })
            .collect::<Vec<_>>();

        let neurons = input_neurons.into_iter().chain(output_neurons).collect();

        Self {
            neurons,
            mutation_chances,
        }
    }

    /// Get the unique identifiers of all neurons in the network.
    ///
    /// # Returns
    /// A vector of UUIDs for all neurons (input, hidden, and output)
    pub fn neuron_ids(&self) -> Vec<Uuid> {
        self.neurons
            .iter()
            .map(|n| n.read().unwrap().id())
            .collect()
    }

    /// Get a reference to all neurons in the network.
    ///
    /// # Returns
    /// A reference to the vector containing all neurons
    pub fn neurons(&self) -> &Vec<Arc<RwLock<PolyNeuronTopology>>> {
        &self.neurons
    }

    pub fn info(&self) -> TopologyInfo {
        let mut info = TopologyInfo::default();
        for neuron in self.neurons.iter() {
            let read_lock = neuron.read().unwrap();
            match read_lock.neuron_type() {
                NeuronType::Input => {
                    info.num_inputs += 1;
                }
                NeuronType::Props(PropsType::Hidden) => {
                    info.num_hidden += 1;
                }
                NeuronType::Props(PropsType::Output) => {
                    info.num_outputs += 1;
                }
            }
        }
        info
    }

    /// Get the mutation configuration for this network.
    ///
    /// # Returns
    /// A reference to the mutation chances configuration
    pub fn mutation_chances(&self) -> &MutationChances {
        &self.mutation_chances
    }

    /// Find a neuron by its unique identifier.
    ///
    /// # Arguments
    /// * `id` - The UUID of the neuron to find
    ///
    /// # Returns
    /// The neuron if found, None otherwise
    pub fn find_by_id(&self, id: Uuid) -> Option<Arc<RwLock<PolyNeuronTopology>>> {
        self.neurons
            .iter()
            .find(|rep| rep.read().unwrap().id() == id)
            .cloned()
    }

    /// Select a random neuron from the network.
    ///
    /// # Arguments
    /// * `rng` - Random number generator for selection
    ///
    /// # Returns
    /// A randomly selected neuron
    pub fn random_neuron(&self, rng: &mut impl Rng) -> &Arc<RwLock<PolyNeuronTopology>> {
        self.neurons
            .get(rng.random_range(0..self.neurons.len()))
            .unwrap()
    }

    /// Remove a random non-input, non-output neuron from the network.
    ///
    /// This method is used during mutation to simplify the network by removing
    /// hidden neurons. Input and output neurons are never removed.
    ///
    /// This action will do nothing if the random selected neuron is an input node's only consumer.
    ///
    /// # Arguments
    /// * `rng` - Random number generator for selection
    pub fn remove_random_neuron(&mut self, rng: &mut impl Rng) {
        if self.neurons.len() > 1 {
            let index = rng.random_range(0..self.neurons.len());

            //let mut input_ids_to_check = HashMap::new();
            {
                // this grabs any inputs to the random index and pushes if the input is an input neuron
                // confusing, but input neurons and hidden neurons can both be inputs.
                let neuron_props = self.neurons.get(index).unwrap().read().unwrap();
                if neuron_props.is_input() || neuron_props.is_output() {
                    return;
                }
                // let Some(props) = neuron_props.props() else {
                //     return;
                // };
                // for input in props.inputs() {
                //     let input = input.input();
                //     let Some(upgrade) = input.upgrade() else {
                //         continue;
                //     };
                //     let Ok(read) = upgrade.read() else {
                //         continue;
                //     };
                //     if read.is_input() {
                //         input_ids_to_check.insert(read.id(), 0);
                //     }
                // }
            }

            // if input_ids_to_check.is_empty() {
            //     self.neurons.remove(index);
            //     return;
            // }

            // // this will iterate through all neurons to count how many nodes are connected to the input neuron
            // // that may be removed.
            // for neuron in &self.neurons {
            //     let lock = neuron.read().unwrap();

            //     let Some(props) = lock.props() else {
            //         //this is an input node if this else branch is hit.
            //         continue;
            //     };
            //     for input in props.inputs() {
            //         let input = input.input();
            //         let Some(upgrade) = input.upgrade() else {
            //             continue;
            //         };
            //         let Ok(neuron_input) = upgrade.read() else {
            //             continue;
            //         };
            //         let Some(count) = input_ids_to_check.get_mut(&neuron_input.id()) else {
            //             continue;
            //         };
            //         *count += 1;
            //     }
            // }

            // // do nothing if removing this node would remove the input from doing anything
            // for count in input_ids_to_check.into_values() {
            //     if count <= 1 {
            //         error!("hit something!");
            //         return;
            //     }
            // }

            self.neurons.remove(index);
        }
    }

    /// Add a new neuron to the network.
    ///
    /// # Arguments
    /// * `neuron` - The neuron to add
    pub fn push(&mut self, neuron: Arc<RwLock<PolyNeuronTopology>>) {
        self.neurons.push(neuron);
    }

    pub fn deep_clone(&self) -> PolyNetworkTopology {
        let mut new_neurons: Vec<Arc<RwLock<PolyNeuronTopology>>> =
            Vec::with_capacity(self.neurons.len());

        // the deep cloning step removes all original inputs for all nodes
        // this needs to happen in its own iteration before updating the nodes for the deep clones
        for neuron in self.neurons.iter() {
            let cloned_neuron = neuron.read().unwrap().deep_clone();

            new_neurons.push(Arc::new(RwLock::new(cloned_neuron)));
        }

        // deep clone the input nodes for the new inputs here
        for (original_neuron, new_neuron) in self.neurons.iter().zip(new_neurons.iter()) {
            let original_neuron = original_neuron.read().unwrap();

            let Some(og_props) = original_neuron.props() else {
                assert!(original_neuron.is_input());
                assert!(new_neuron.read().unwrap().is_input());
                continue;
            };

            let mut cloned_inputs: Vec<PolyInputTopology> =
                Vec::with_capacity(og_props.inputs().len());

            for og_input in og_props.inputs() {
                if let Some(strong_parent) = og_input.neuron() {
                    if let Some(index) = self
                        .neurons
                        .iter()
                        .position(|n| Arc::ptr_eq(n, &strong_parent))
                    {
                        let cloned_ident_ref = Arc::downgrade(&new_neurons[index]);

                        let cloned_input_topology = PolyInputTopology::new(
                            cloned_ident_ref,
                            og_input.weight(),
                            og_input.exponent(),
                        );

                        cloned_inputs.push(cloned_input_topology);
                    }
                }
            }

            // inputs should be fully cloned at this point
            match new_neuron.write().unwrap().props_mut() {
                Some(props_mut) => props_mut.set_inputs(cloned_inputs),
                None => {
                    unreachable!("this check should be invalid due to the check on the input type")
                }
            }
        }

        PolyNetworkTopology {
            neurons: new_neurons,
            mutation_chances: self.mutation_chances,
        }
    }

    //#[instrument(skip_all)]
    pub fn replicate(&self, rng: &mut impl Rng) -> PolyNetworkTopology {
        let mut child = self.deep_clone();

        let actions = self.mutation_chances.gen_mutation_actions(rng);
        child.mutate(actions.as_slice(), rng);

        child.mutation_chances.adjust_mutation_chances(rng);

        child.remove_cycles();

        child
    }

    #[cfg(test)]
    /// Generate a human-readable debug string representation of the network.
    ///
    /// This provides a detailed view of the network structure including:
    /// - All neurons and their types
    /// - Connection patterns between neurons
    /// - Input and output layer composition
    ///
    /// Useful for debugging and visualizing network architectures.
    ///
    /// # Returns
    /// A formatted string describing the network structure
    pub fn debug_str(&self) -> String {
        let mut str = String::new();
        for (neuron_index, neuron) in self.neurons.iter().enumerate() {
            let neuron = neuron.read().unwrap();
            str.push_str(&format!(
                "\n(({}) {}[{}]: ",
                neuron_index,
                neuron.id_short(),
                neuron.neuron_type()
            ));
            match neuron.props() {
                Some(props) => {
                    str.push('[');
                    for input in props.inputs() {
                        match input.neuron() {
                            Some(n) => {
                                let n = n.read().unwrap();

                                let loc = self
                                    .neurons
                                    .iter()
                                    .position(|neuron| neuron.read().unwrap().id() == n.id())
                                    .unwrap();

                                str.push_str(&format!("({})", loc));
                            }
                            None => str.push_str("(DROPPED)"),
                        }
                    }
                    str.push(']')
                }

                None => {
                    str.push_str("N/A");
                }
            }

            str.push(')');
        }
        str
    }

    pub fn mutate(&mut self, actions: &[MutationAction], rng: &mut impl Rng) {
        use MutationAction::*;

        for action in actions {
            match action {
                SplitConnection => {
                    // clone the arc to borrow later
                    let neuron_to_split = Arc::clone(self.random_neuron(rng));
                    let removed_input = match neuron_to_split.write().unwrap().props_mut() {
                        Some(props) => props.remove_random_input(rng),
                        None => None,
                    };

                    let Some(removed_input) = removed_input else {
                        continue;
                    };

                    //make a new neuron
                    let new_hidden_node = Arc::new(RwLock::new(PolyNeuronTopology::hidden(
                        Uuid::new_v4(),
                        vec![removed_input],
                    )));

                    self.push(Arc::clone(&new_hidden_node));

                    //add the new hidden node to the list of inputs for the neuron
                    let new_replicant_for_neuron = PolyInputTopology::new(
                        Arc::downgrade(&new_hidden_node),
                        Bias::rand(rng),
                        Exponent::rand(rng),
                    );

                    let mut neuron_to_split = neuron_to_split.write().unwrap();

                    //If the arc is removed from the array at this point, it will disappear, and the weak reference will
                    //ultimately be removed.
                    if let Some(props) = neuron_to_split.props_mut() {
                        props.add_input(new_replicant_for_neuron);
                    }
                }
                AddConnection => {
                    // the input neuron gets added to the output neuron's list of inputs
                    let output_neuron = self.random_neuron(rng);
                    let input_neuron = self.random_neuron(rng);

                    //the input neuron cannot be an output and the output cannot be an input.
                    if input_neuron.read().unwrap().is_output() {
                        continue;
                    }

                    if let Some(props) = output_neuron.write().unwrap().props_mut() {
                        let input = PolyInputTopology::new(
                            Arc::downgrade(input_neuron),
                            Bias::rand(rng),
                            Exponent::rand(rng),
                        );
                        props.add_input(input);
                    }
                }
                RemoveNeuron => {
                    // remove a random neuron, if it has any.
                    self.remove_random_neuron(rng);
                }
                MutateWeight => {
                    let mut neuron = self.random_neuron(rng).write().unwrap();
                    let Some(random_input) = neuron
                        .props_mut()
                        .and_then(|props| props.get_random_input_mut(rng))
                    else {
                        continue;
                    };

                    random_input.adjust_weight(rng.random_range(-1.0..=1.0));
                }
                MutateExponent => {
                    let mut neuron = self.random_neuron(rng).write().unwrap();
                    let Some(random_input) = neuron
                        .props_mut()
                        .and_then(|props| props.get_random_input_mut(rng))
                    else {
                        continue;
                    };

                    random_input.adjust_exp(rng.random_range(-1..=1));
                    // let adjustment = rng.random_range(-1..=1);
                    // let new_exp = random_input.exponent() + adjustment;
                    // // Ensure exponent stays non-negative to avoid division by zero
                    // if new_exp >= 0 {
                    //     random_input.adjust_exp(adjustment);
                    // }
                }
            }
        }
    }

    /// Remove cycles from the network to ensure it remains feedforward.
    ///
    /// This method uses depth-first search to detect and remove connections
    /// that would create cycles in the network. This ensures the network
    /// can be evaluated in a single forward pass without infinite loops.
    ///
    /// Cycles are removed by disconnecting neurons from their cyclic inputs.
    fn remove_cycles(&mut self) {
        let mut stack = HashSet::new();
        let mut visited = HashSet::new();

        #[derive(Debug)]
        struct RemoveFrom {
            remove_from: Uuid,
            indices: Vec<usize>,
        }

        fn dfs(
            node: &PolyNeuronTopology,
            stack: &mut HashSet<Uuid>,
            visited: &mut HashSet<Uuid>,
        ) -> Vec<RemoveFrom> {
            let node_id = node.id();
            visited.insert(node_id);

            match node.props().map(|props| props.inputs()) {
                Some(inputs) => {
                    stack.insert(node_id);

                    let mut total_remove = Vec::new();
                    let mut self_remove_indices = Vec::new();
                    for (input_indice, input) in inputs.iter().enumerate() {
                        let Some(input_neuron) = input.neuron() else {
                            continue;
                        };
                        let input_neuron_id = input_neuron.read().unwrap().id();

                        if !visited.contains(&input_neuron_id) {
                            let child_result = dfs(&input_neuron.read().unwrap(), stack, visited);
                            if !child_result.is_empty() {
                                total_remove.extend(child_result);
                            }
                        } else if stack.contains(&input_neuron_id) {
                            self_remove_indices.push(input_indice);
                        }
                    }

                    if !self_remove_indices.is_empty() {
                        total_remove.push(RemoveFrom {
                            remove_from: node_id,
                            indices: self_remove_indices,
                        });
                    }

                    stack.remove(&node_id);
                    total_remove
                }
                None => vec![],
            }
        }
        let mut _num_removed = 0;
        loop {
            let mut remove_queue = Vec::new();

            for neuron in self.neurons.iter() {
                let id = neuron.read().unwrap().id();

                if visited.contains(&id) {
                    continue;
                }

                let to_remove = dfs(&neuron.read().unwrap(), &mut stack, &mut visited);

                if !to_remove.is_empty() {
                    remove_queue = to_remove;
                    break;
                }
            }
            if remove_queue.is_empty() {
                break;
            }
            for removal in remove_queue {
                let neuron_to_trim = self
                    .neurons
                    .iter_mut()
                    .find(|neuron| neuron.read().unwrap().id() == removal.remove_from)
                    .unwrap();
                let mut neuron = neuron_to_trim.write().unwrap();
                let Some(props) = neuron.props_mut() else {
                    panic!("tried to remove inputs from an input node!");
                };
                props.trim_inputs(removal.indices.as_slice());
                _num_removed += 1;
            }
        }

        //info!("Num removed: {}", num_removed);
        /*
        neuron.write().unwrap().trim_inputs(to_remove);*/
    }

    //#[instrument(name = "my_span")]
    pub fn to_simple_network(&self) -> SimplePolyNetwork {
        SimplePolyNetwork::from_topology(self)
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct TopologyInfo {
    pub num_inputs: usize,
    pub num_hidden: usize,
    pub num_outputs: usize,
}

#[test]
fn make_simple_network() {
    let input = arc(PolyNeuronTopology::input(Uuid::new_v4()));

    let hidden_1 = arc(PolyNeuronTopology::hidden(
        Uuid::new_v4(),
        vec![
            PolyInputTopology::downgrade(&input, 3., 1),
            PolyInputTopology::downgrade(&input, 1., 2),
        ],
    ));

    let hidden_2 = arc(PolyNeuronTopology::hidden(
        Uuid::new_v4(),
        vec![PolyInputTopology::downgrade(&input, 1., 2)],
    ));

    let output = arc(PolyNeuronTopology::output(
        Uuid::new_v4(),
        vec![
            PolyInputTopology::downgrade(&hidden_1, 1., 1),
            PolyInputTopology::downgrade(&hidden_2, 1., 1),
        ],
    ));

    let topology = PolyNetworkTopology::from_raw_parts(
        vec![input, hidden_1, hidden_2, output],
        MutationChances::none(),
    );

    assert_eq!(topology.neurons().len(), 4);
    assert_eq!(*topology.mutation_chances(), MutationChances::none());
}
