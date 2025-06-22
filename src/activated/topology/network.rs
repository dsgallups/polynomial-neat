use std::{
    collections::HashSet,
    sync::{Arc, RwLock},
};

use rand::Rng;
use uuid::Uuid;

use crate::activated::prelude::*;

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
            .map(|_| NeuronTopology::input(Uuid::new_v4()))
            .collect::<Vec<_>>();

        let output_neurons = (0..num_outputs)
            .map(|_| {
                //a random number of connections to random input neurons;
                let mut chosen_inputs = (0..rng.random_range(1..input_neurons.len()))
                    .map(|_| {
                        let topology_index = rng.random_range(0..input_neurons.len());
                        let input = input_neurons.get(topology_index).unwrap();
                        (
                            InputTopology::new_rand(Arc::downgrade(input), rng),
                            topology_index,
                        )
                    })
                    .collect::<Vec<_>>();

                chosen_inputs.sort_by_key(|(_, i)| *i);
                chosen_inputs.dedup_by_key(|(_, i)| *i);

                let chosen_inputs = chosen_inputs.into_iter().map(|(input, _)| input).collect();

                NeuronTopology::output_rand(chosen_inputs, rng)
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
            .map(|_| NeuronTopology::input(Uuid::new_v4()))
            .collect::<Vec<_>>();

        let output_neurons = (0..num_outputs)
            .map(|_| {
                //every output neuron is connected to every input neuron

                let chosen_inputs = input_neurons
                    .iter()
                    .map(|input| InputTopology::new_rand(Arc::downgrade(input), rng))
                    .collect::<Vec<_>>();

                NeuronTopology::output_rand(chosen_inputs, rng)
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
            .get(rng.random_range(0..self.neurons.len()))
            .unwrap()
    }
    pub fn remove_random_neuron(&mut self, rng: &mut impl Rng) {
        if self.neurons.len() > 1 {
            let index = rng.random_range(0..self.neurons.len());

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
        for neuron in self.neurons.iter() {
            let cloned_neuron = neuron.read().unwrap().deep_clone();

            new_neurons.push(Arc::new(RwLock::new(cloned_neuron)));
        }

        // deep clone the input nodes for the new inputs here
        for (original_neuron, new_neuron) in self.neurons.iter().zip(new_neurons.iter()) {
            let original_neuron = original_neuron.read().unwrap();
            let Some(og_inputs) = original_neuron.inputs() else {
                assert!(original_neuron.is_input());
                assert!(new_neuron.read().unwrap().is_input());
                continue;
            };

            let mut cloned_inputs: Vec<InputTopology> = Vec::with_capacity(og_inputs.len());

            for og_input in og_inputs {
                if let Some(strong_parent) = og_input.neuron() {
                    if let Some(index) = self
                        .neurons
                        .iter()
                        .position(|n| Arc::ptr_eq(n, &strong_parent))
                    {
                        let cloned_ident_ref = Arc::downgrade(&new_neurons[index]);

                        let cloned_input_topology =
                            InputTopology::new(cloned_ident_ref, og_input.weight());

                        cloned_inputs.push(cloned_input_topology);
                    }
                }
            }

            // inputs should be fully cloned at this point
            new_neuron.write().unwrap().set_inputs(cloned_inputs);
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

    pub fn mutate(&mut self, actions: &[MutationAction], rng: &mut impl Rng) {
        use MutationAction::*;

        for action in actions {
            match action {
                SplitConnection => {
                    // clone the arc to borrow later
                    let neuron_to_split = Arc::clone(self.random_neuron(rng));
                    let removed_input = {
                        let mut neuron_to_split = neuron_to_split.write().unwrap();
                        neuron_to_split.remove_random_input(rng)
                    };
                    let Some(removed_input) = removed_input else {
                        continue;
                    };

                    //make a new neuron
                    let new_hidden_node = NeuronTopology::hidden_rand(vec![removed_input], rng);

                    self.push(Arc::clone(&new_hidden_node));

                    //add the new hidden node to the list of inputs for the neuron
                    let new_replicant_for_neuron =
                        InputTopology::new(Arc::downgrade(&new_hidden_node), Bias::rand(rng));

                    let mut neuron_to_split = neuron_to_split.write().unwrap();
                    neuron_to_split.add_input(new_replicant_for_neuron);
                    //If the arc is removed from the array at this point, it will disappear, and the weak reference will
                    //ultimately be removed.
                }
                AddConnection => {
                    // the input neuron gets added to the output neuron's list of inputs
                    let output_neuron = self.random_neuron(rng);
                    let input_neuron = self.random_neuron(rng);

                    //the input neuron cannot be an output and the output cannot be an input.
                    if input_neuron.read().unwrap().is_output()
                        || output_neuron.read().unwrap().is_input()
                    {
                        continue;
                    }

                    let mut output_neuron = output_neuron.write().unwrap();
                    let input = InputTopology::new(Arc::downgrade(input_neuron), Bias::rand(rng));
                    output_neuron.add_input(input);
                }
                RemoveNeuron => {
                    // remove a random neuron, if it has any.
                    self.remove_random_neuron(rng);
                }
                MutateWeight => {
                    let mut neuron = self.random_neuron(rng).write().unwrap();
                    let Some(random_input) = neuron.get_random_input_mut(rng) else {
                        continue;
                    };
                    random_input.adjust_weight(rng.random_range(-1.0..=1.0));
                }
                MutateActivationFunction => {
                    let mut neuron = self.random_neuron(rng).write().unwrap();
                    let Some(activation) = neuron.activation_mut() else {
                        continue;
                    };
                    *activation = Activation::rand(rng);
                }
                MutateBias => {
                    let mut neuron = self.random_neuron(rng).write().unwrap();
                    let Some(bias) = neuron.bias_mut() else {
                        continue;
                    };
                    *bias += rng.random_range(-1.0..=1.0);
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
            stack.insert(node_id);
            visited.insert(node_id);

            let mut total_remove = Vec::new();

            if let Some(inputs) = node.inputs() {
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
            }

            stack.remove(&node_id);
            total_remove
        }
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
            for neuron in self.neurons.iter() {
                let id = neuron.read().unwrap().id();

                if let Some(remove) = remove_queue.iter().find(|r| r.remove_from == id) {
                    let mut write_lock = neuron.write().unwrap();
                    write_lock.trim_inputs(remove.indices.as_slice());
                }
            }
        }
        /*
        neuron.write().unwrap().trim_inputs(to_remove);*/
    }

    //#[instrument(name = "my_span")]
    pub fn to_network(&self) -> Network {
        let mut neurons: Vec<Arc<RwLock<Neuron>>> = Vec::with_capacity(self.neurons.len());
        let mut input_layer: Vec<Arc<RwLock<Neuron>>> = Vec::new();
        let mut output_layer: Vec<Arc<RwLock<Neuron>>> = Vec::new();

        for neuron_replicant in self.neurons.iter() {
            let neuron = neuron_replicant.read().unwrap();
            let neuron = neuron.to_neuron(&mut neurons, &self.neurons);
            neurons.push(Arc::clone(&neuron));
            let neuron_read = neuron.read().unwrap();
            if neuron_read.is_input() {
                input_layer.push(Arc::clone(&neuron));
            }
            if neuron_read.is_output() {
                output_layer.push(Arc::clone(&neuron));
            }
        }

        Network::from_raw_parts(neurons, input_layer, output_layer)
    }
}
