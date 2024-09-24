use std::{
    collections::HashSet,
    sync::{Arc, RwLock},
};

use rand::Rng;
use tracing::info;
use uuid::Uuid;

use crate::{prelude::*, topology::activation::Exponent};

use super::mutation::MutationChances;

#[derive(Clone, Debug)]
pub struct NetworkTopology {
    neurons: Vec<Arc<RwLock<NeuronTopology>>>,
    mutation_chances: MutationChances,
}

impl NetworkTopology {
    pub fn new(
        num_inputs: usize,
        num_outputs: usize,
        mutation_chances: MutationChances,
        rng: &mut impl Rng,
    ) -> Self {
        let input_neurons = (0..num_inputs)
            .map(|_| Arc::new(RwLock::new(NeuronTopology::input(Uuid::new_v4()))))
            .collect::<Vec<_>>();

        let output_neurons = (0..num_outputs)
            .map(|_| {
                //a random number of connections to random input neurons;
                let mut chosen_inputs = (0..rng.gen_range(1..input_neurons.len()))
                    .map(|_| {
                        let topology_index = rng.gen_range(0..input_neurons.len());
                        let input = input_neurons.get(topology_index).unwrap();
                        (
                            InputTopology::new_rand(Arc::downgrade(input), &mut rand::thread_rng()),
                            topology_index,
                        )
                    })
                    .collect::<Vec<_>>();

                chosen_inputs.sort_by_key(|(_, i)| *i);
                chosen_inputs.dedup_by_key(|(_, i)| *i);

                let chosen_inputs = chosen_inputs.into_iter().map(|(input, _)| input).collect();

                Arc::new(RwLock::new(NeuronTopology::output(
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
            .map(|_| Arc::new(RwLock::new(NeuronTopology::input(Uuid::new_v4()))))
            .collect::<Vec<_>>();

        let output_neurons = (0..num_outputs)
            .map(|_| {
                //every output neuron is connected to every input neuron

                let chosen_inputs = input_neurons
                    .iter()
                    .map(|input| InputTopology::new_rand(Arc::downgrade(input), rng))
                    .collect::<Vec<_>>();

                Arc::new(RwLock::new(NeuronTopology::output(
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

    pub fn neuron_ids(&self) -> Vec<Uuid> {
        self.neurons
            .iter()
            .map(|n| n.read().unwrap().id())
            .collect()
    }

    pub fn mutation_chances(&self) -> &MutationChances {
        &self.mutation_chances
    }

    pub fn find_by_id(&self, id: Uuid) -> Option<&Arc<RwLock<NeuronTopology>>> {
        self.neurons
            .iter()
            .find(|rep| rep.read().unwrap().id() == id)
    }

    pub fn random_neuron(&self, rng: &mut impl Rng) -> &Arc<RwLock<NeuronTopology>> {
        self.neurons
            .get(rng.gen_range(0..self.neurons.len()))
            .unwrap()
    }
    pub fn remove_random_neuron(&mut self, rng: &mut impl Rng) {
        if self.neurons.len() > 1 {
            let index = rng.gen_range(0..self.neurons.len());

            {
                let neuron_props = self.neurons.get(index).unwrap().read().unwrap();
                if neuron_props.is_input() || neuron_props.is_output() {
                    return;
                }
            }

            self.neurons.remove(index);
        }
    }

    pub fn push(&mut self, rep: Arc<RwLock<NeuronTopology>>) {
        self.neurons.push(rep);
    }

    pub fn deep_clone(&self) -> NetworkTopology {
        let mut new_neurons: Vec<Arc<RwLock<NeuronTopology>>> =
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

            let mut cloned_inputs: Vec<InputTopology> = Vec::with_capacity(og_props.inputs().len());

            for og_input in og_props.inputs() {
                if let Some(strong_parent) = og_input.neuron() {
                    if let Some(index) = self
                        .neurons
                        .iter()
                        .position(|n| Arc::ptr_eq(n, &strong_parent))
                    {
                        let cloned_ident_ref = Arc::downgrade(&new_neurons[index]);

                        let cloned_input_topology = InputTopology::new(
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

        NetworkTopology {
            neurons: new_neurons,
            mutation_chances: self.mutation_chances,
        }
    }

    //#[instrument(skip_all)]
    pub fn replicate(&self, rng: &mut impl Rng) -> NetworkTopology {
        let mut child = self.deep_clone();

        let actions = self.mutation_chances.gen_mutation_actions(rng);
        child.mutate(actions.as_slice(), rng);

        child.mutation_chances.adjust_mutation_chances(rng);

        child.remove_cycles();

        child
    }

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
                    let new_hidden_node = Arc::new(RwLock::new(NeuronTopology::hidden(
                        Uuid::new_v4(),
                        vec![removed_input],
                    )));

                    self.push(Arc::clone(&new_hidden_node));

                    //add the new hidden node to the list of inputs for the neuron
                    let new_replicant_for_neuron = InputTopology::new(
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
                        let input = InputTopology::new(
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

                    random_input.adjust_weight(rng.gen_range(-1.0..=1.0));
                }
                MutateExponent => {
                    let mut neuron = self.random_neuron(rng).write().unwrap();
                    let Some(random_input) = neuron
                        .props_mut()
                        .and_then(|props| props.get_random_input_mut(rng))
                    else {
                        continue;
                    };
                    random_input.adjust_exp(rng.gen_range(-1..=1));
                }
            }
        }
    }

    fn remove_cycles(&mut self) {
        let mut stack = HashSet::new();
        let mut visited = HashSet::new();

        #[derive(Debug)]
        struct RemoveFrom {
            remove_from: Uuid,
            indices: Vec<usize>,
        }

        fn dfs(
            node: &NeuronTopology,
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
        let mut num_removed = 0;
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
                num_removed += 1;
            }
        }

        info!("Num removed: {}", num_removed);
        /*
        neuron.write().unwrap().trim_inputs(to_remove);*/
    }

    //#[instrument(name = "my_span")]
    pub fn to_simple_network(&self) -> SimpleNetwork {
        let mut neurons: Vec<Arc<RwLock<Neuron>>> = Vec::with_capacity(self.neurons.len());
        let mut input_layer: Vec<Arc<RwLock<Neuron>>> = Vec::new();
        let mut output_layer: Vec<Arc<RwLock<Neuron>>> = Vec::new();

        for neuron_replicant in self.neurons.iter() {
            let neuron = neuron_replicant.read().unwrap();

            neuron.to_neuron(&mut neurons);
            let neuron = neurons
                .iter()
                .find(|n| n.read().unwrap().id() == neuron.id())
                .unwrap();

            let neuron_read = neuron.read().unwrap();

            if neuron_read.is_input() {
                input_layer.push(Arc::clone(neuron));
            }
            if neuron_read.is_output() {
                output_layer.push(Arc::clone(neuron));
            }
        }

        info!(
            "Network final: ({}, {}, {})",
            neurons.len(),
            input_layer.len(),
            output_layer.len()
        );

        SimpleNetwork::from_raw_parts(neurons, input_layer, output_layer)
    }
}
