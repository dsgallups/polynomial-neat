use std::{
    collections::HashSet,
    hint::unreachable_unchecked,
    sync::{Arc, RwLock, Weak},
};

use rand::Rng;
use uuid::Uuid;

use crate::{
    prelude::*,
    topology::{activation::Bias, neuron::NeuronInputTopology},
};

pub enum MutationAction {
    SplitConnection,
    Add,
    Remove,
    MutateWeight,
    MutateBias,
    MutateActivationFunction,
}

trait MutationRateExt {
    fn gen_rate(&mut self) -> u8;

    fn gen_mutation_action(&mut self) -> MutationAction;
}

impl<T: Rng> MutationRateExt for T {
    fn gen_rate(&mut self) -> u8 {
        self.gen_range(0..=100)
    }

    fn gen_mutation_action(&mut self) -> MutationAction {
        use MutationAction::*;
        match self.gen_range(0..6) {
            0 => SplitConnection,
            1 => Add,
            2 => Remove,
            3 => MutateWeight,
            4 => MutateBias,
            5 => MutateActivationFunction,
            // Safety: Cannot generate a value more than 5
            _ => unsafe { unreachable_unchecked() },
        }
    }
}

const MAX_MUTATIONS: u8 = 200;

#[derive(Clone)]
pub struct NeuronReplicants(Vec<Arc<RwLock<NeuronReplicant>>>);

impl NeuronReplicants {
    pub fn with_capacity(cap: usize) -> Self {
        Self(Vec::with_capacity(cap))
    }

    pub fn find_by_id(&self, id: Uuid) -> Option<&Arc<RwLock<NeuronReplicant>>> {
        self.0.iter().find(|rep| rep.read().unwrap().id == id)
    }

    pub fn random_neuron(&self, rng: &mut impl Rng) -> &Arc<RwLock<NeuronReplicant>> {
        self.0.get(rng.gen_range(0..self.0.len())).unwrap()
    }

    pub fn push(&mut self, rep: Arc<RwLock<NeuronReplicant>>) {
        self.0.push(rep);
    }

    pub fn mutate(&mut self, rate: u8, rng: &mut impl Rng) {
        use MutationAction::*;
        let mut mutation_count = 0;

        while rng.gen_rate() <= rate && mutation_count < MAX_MUTATIONS {
            match rng.gen_mutation_action() {
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
                    let new_hidden_node = NeuronReplicant::hidden_rand(vec![removed_input], rng);

                    self.push(Arc::clone(&new_hidden_node));

                    //add the new hidden node to the list of inputs for the neuron
                    let new_replicant_for_neuron =
                        InputReplicant::new(Arc::downgrade(&new_hidden_node), Bias::rand(rng));

                    let mut neuron_to_split = neuron_to_split.write().unwrap();
                    neuron_to_split.add_input(new_replicant_for_neuron);
                    //If the arc is removed from the array at this point, it will disappear, and the weak reference will
                    //ultimately be removed.
                }
                Add => {
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
                    let input = InputReplicant::new(Arc::downgrade(input_neuron), Bias::rand(rng));
                    output_neuron.add_input(input);
                }
                Remove => {
                    // remove a random input node, if it has any.
                    let remove_from = self.random_neuron(rng);
                    let mut remove_from = remove_from.write().unwrap();
                    remove_from.remove_random_input(rng);
                }
                MutateWeight => {
                    let mut neuron = self.random_neuron(rng).write().unwrap();
                    let Some(random_input) = neuron.get_random_input_mut(rng) else {
                        continue;
                    };
                    random_input.weight += rng.gen_range(-1.0..=1.0);
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
                    *bias += rng.gen_range(-1.0..=1.0);
                }
            }

            mutation_count += 1;
        }
    }

    pub fn remove_cycles(&mut self) {
        let mut visited = HashSet::new();
        let mut stack = HashSet::new();

        fn dfs(node: &mut NeuronReplicant, visited: &mut HashSet<Uuid>, stack: &mut HashSet<Uuid>) {
            visited.insert(node.id());
            stack.insert(node.id());

            let ids_to_remove = if let Some(inputs) = node.inputs() {
                let mut to_remove: Vec<Uuid> = Vec::new();
                for input in inputs {
                    let Some(input_neuron) = input.neuron() else {
                        // neuron has been dropped already
                        continue;
                    };
                    let input_neuron_id = input_neuron.read().unwrap().id();
                    if !visited.contains(&input_neuron_id) {
                        dfs(&mut input_neuron.write().unwrap(), visited, stack);
                    } else if stack.contains(&input_neuron_id) {
                        to_remove.push(input_neuron_id);
                    }
                }
                to_remove
            } else {
                vec![]
            };

            node.trim_inputs(ids_to_remove);

            stack.remove(&node.id());
        }

        for replicant in self.0.iter() {
            let replicant_id = replicant.read().unwrap().id();
            if !visited.contains(&replicant_id) {
                dfs(&mut replicant.write().unwrap(), &mut visited, &mut stack);
            }
        }
    }

    pub fn into_neuron_topology(mut self) -> Vec<NeuronTopology> {
        self.remove_cycles();

        let mut neuron_topology = Vec::new();

        for replicant in self.0.iter() {
            let replicant = replicant.read().unwrap();
            replicant.to_topology(&mut neuron_topology);
        }
        neuron_topology
    }

    pub fn to_network(&self) -> Network {
        let mut neurons: Vec<Arc<RwLock<Neuron>>> = Vec::with_capacity(self.0.len());
        let mut input_layer: Vec<Arc<RwLock<Neuron>>> = Vec::new();
        let mut output_layer: Vec<Arc<RwLock<Neuron>>> = Vec::new();

        for neuron_replicant in self.0.iter() {
            let neuron = neuron_replicant.read().unwrap();
            let neuron = neuron.to_neuron(&mut neurons, &self.0);
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

#[derive(Clone)]
pub struct InputReplicant {
    input: Weak<RwLock<NeuronReplicant>>,
    weight: f32,
}

impl InputReplicant {
    pub fn new(input: Weak<RwLock<NeuronReplicant>>, weight: f32) -> Self {
        Self { input, weight }
    }

    pub fn neuron(&self) -> Option<Arc<RwLock<NeuronReplicant>>> {
        Weak::upgrade(&self.input)
    }
    pub fn weight(&self) -> f32 {
        self.weight
    }
}

#[derive(Clone)]
pub enum NeuronTypeReplicant {
    Input,
    Hidden {
        inputs: Vec<InputReplicant>,
        activation: Activation,
        bias: f32,
    },
    Output {
        inputs: Vec<InputReplicant>,
        activation: Activation,
        bias: f32,
    },
}

impl NeuronTypeReplicant {
    pub fn input() -> Self {
        Self::Input
    }
    pub fn hidden(inputs: Vec<InputReplicant>, activation: Activation, bias: f32) -> Self {
        Self::Hidden {
            inputs,
            activation,
            bias,
        }
    }
    pub fn output(inputs: Vec<InputReplicant>, activation: Activation, bias: f32) -> Self {
        Self::Output {
            inputs,
            activation,
            bias,
        }
    }

    pub fn add_input(&mut self, input: InputReplicant) {
        use NeuronTypeReplicant::*;
        match self {
            Input => panic!("Cannot add input to an input node!"),
            Hidden { ref mut inputs, .. } | Output { ref mut inputs, .. } => {
                inputs.push(input);
            }
        }
    }
    pub fn inputs(&self) -> Option<&[InputReplicant]> {
        use NeuronTypeReplicant::*;
        match self {
            Input => None,
            Hidden { ref inputs, .. } | Output { ref inputs, .. } => Some(inputs),
        }
    }

    // Clears out all inputs whose reference is dropped or match on the provided ids
    pub fn trim_inputs(&mut self, ids: Vec<Uuid>) {
        use NeuronTypeReplicant::*;
        match self {
            Input => {}
            Hidden { ref mut inputs, .. } | Output { ref mut inputs, .. } => {
                inputs.retain(|input| {
                    let Some(input) = input.neuron() else {
                        return false;
                    };
                    let input = input.read().unwrap();

                    !ids.contains(&input.id())
                });
            }
        }
    }

    /// Returnes the removed input, if it has inputs.
    pub fn remove_random_input(&mut self, rng: &mut impl Rng) -> Option<InputReplicant> {
        use NeuronTypeReplicant::*;
        match self {
            Input => None,
            Hidden { inputs, .. } | Output { inputs, .. } => {
                if inputs.is_empty() {
                    return None;
                }
                let removed = inputs.swap_remove(rng.gen_range(0..inputs.len()));
                Some(removed)
            }
        }
    }

    pub fn get_random_input_mut(&mut self, rng: &mut impl Rng) -> Option<&mut InputReplicant> {
        use NeuronTypeReplicant::*;
        match self {
            Input => None,
            Hidden { inputs, .. } | Output { inputs, .. } => {
                if inputs.is_empty() {
                    return None;
                }
                let len = inputs.len();
                inputs.get_mut(rng.gen_range(0..len))
            }
        }
    }
    pub fn activation(&self) -> Option<Activation> {
        use NeuronTypeReplicant::*;
        match self {
            Input => None,
            Hidden { activation, .. } | Output { activation, .. } => Some(*activation),
        }
    }
    pub fn activation_mut(&mut self) -> Option<&mut Activation> {
        use NeuronTypeReplicant::*;
        match self {
            Input => None,
            Hidden { activation, .. } | Output { activation, .. } => Some(activation),
        }
    }
    pub fn bias(&self) -> Option<f32> {
        use NeuronTypeReplicant::*;
        match self {
            Input => None,
            Hidden { bias, .. } | Output { bias, .. } => Some(*bias),
        }
    }
    pub fn bias_mut(&mut self) -> Option<&mut f32> {
        use NeuronTypeReplicant::*;
        match self {
            Input => None,
            Hidden { bias, .. } | Output { bias, .. } => Some(bias),
        }
    }

    pub fn num_inputs(&self) -> usize {
        use NeuronTypeReplicant::*;
        match self {
            Input => 0,
            Hidden { inputs, .. } | Output { inputs, .. } => inputs.len(),
        }
    }

    pub fn is_output(&self) -> bool {
        matches!(self, Self::Output { .. })
    }
    pub fn is_input(&self) -> bool {
        matches!(self, Self::Input)
    }
    pub fn is_hidden(&self) -> bool {
        matches!(self, Self::Hidden { .. })
    }
}

#[derive(Clone)]
pub struct NeuronReplicant {
    id: Uuid,
    neuron_type: NeuronTypeReplicant,
}

impl NeuronReplicant {
    pub fn input(id: Uuid) -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(Self {
            id,
            neuron_type: NeuronTypeReplicant::input(),
        }))
    }

    pub fn new(id: Uuid, neuron_type: NeuronTypeReplicant) -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(Self { id, neuron_type }))
    }

    pub fn hidden_rand(inputs: Vec<InputReplicant>, rng: &mut impl Rng) -> Arc<RwLock<Self>> {
        let id = Uuid::new_v4();
        let neuron_type =
            NeuronTypeReplicant::hidden(inputs, Activation::rand(rng), Bias::rand(rng));
        Self::new(id, neuron_type)
    }

    /// returns the index of the neuron in the vector
    pub fn to_topology(&self, neuron_topology: &mut Vec<NeuronTopology>) -> usize {
        if let Some(index) = neuron_topology
            .iter()
            .position(|neuron| neuron.id() == self.id())
        {
            return index;
        }

        let neuron_type = if let Some(inputs) = self.inputs() {
            let mut new_inputs: Vec<NeuronInputTopology> = Vec::with_capacity(self.num_inputs());
            for input in inputs {
                if let Some(input_node) = input.neuron() {
                    let index = input_node.read().unwrap().to_topology(neuron_topology);
                    let input_topology = NeuronInputTopology::new(index, input.weight());
                    new_inputs.push(input_topology);
                }
            }
            if self.is_hidden() {
                NeuronTopologyType::hidden(
                    new_inputs,
                    self.activation().unwrap(),
                    self.bias().unwrap(),
                )
            } else {
                //is output
                NeuronTopologyType::output(
                    new_inputs,
                    self.activation().unwrap(),
                    self.bias().unwrap(),
                )
            }
        } else {
            NeuronTopologyType::Input
        };

        neuron_topology.push(NeuronTopology::new(self.id(), neuron_type));
        neuron_topology.len()
    }

    pub fn from_topology(
        neuron: &NeuronTopology,
        replicants: &mut NeuronReplicants,
        parent_neurons: &[NeuronTopology],
    ) -> Arc<RwLock<Self>> {
        let id = neuron.id();
        if let Some(replicant) = replicants.find_by_id(id) {
            return Arc::clone(replicant);
        }

        //not found
        match neuron.inputs() {
            None => {
                let val = NeuronReplicant::input(id);
                replicants.push(Arc::clone(&val));
                val
            }
            Some(inputs) => {
                let mut new_inputs = Vec::with_capacity(inputs.len());
                for input in inputs {
                    let input_neuron_top = parent_neurons.get(input.topology_index()).unwrap();
                    let neuron_replicant = match replicants.find_by_id(input_neuron_top.id()) {
                        Some(replicant) => Arc::downgrade(replicant),
                        None => Arc::downgrade(&NeuronReplicant::from_topology(
                            input_neuron_top,
                            replicants,
                            parent_neurons,
                        )),
                    };
                    let input_replicant = InputReplicant::new(neuron_replicant, input.weight());
                    new_inputs.push(input_replicant);
                }

                let neuron_type = if neuron.is_hidden() {
                    NeuronTypeReplicant::hidden(
                        new_inputs,
                        neuron.activation().unwrap(),
                        neuron.bias().unwrap(),
                    )
                } else {
                    NeuronTypeReplicant::output(
                        new_inputs,
                        neuron.activation().unwrap(),
                        neuron.bias().unwrap(),
                    )
                };
                let val = NeuronReplicant::new(id, neuron_type);
                replicants.push(Arc::clone(&val));
                val
            }
        }
    }

    pub fn get_random_input_mut(&mut self, rng: &mut impl Rng) -> Option<&mut InputReplicant> {
        self.neuron_type.get_random_input_mut(rng)
    }

    pub fn activation(&self) -> Option<Activation> {
        self.neuron_type.activation()
    }

    pub fn activation_mut(&mut self) -> Option<&mut Activation> {
        self.neuron_type.activation_mut()
    }
    pub fn bias(&self) -> Option<f32> {
        self.neuron_type.bias()
    }
    pub fn bias_mut(&mut self) -> Option<&mut f32> {
        self.neuron_type.bias_mut()
    }
    pub fn num_inputs(&self) -> usize {
        self.neuron_type.num_inputs()
    }

    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Returnes the removed input, if it has inputs.
    pub fn remove_random_input(&mut self, rng: &mut impl Rng) -> Option<InputReplicant> {
        self.neuron_type.remove_random_input(rng)
    }

    pub fn add_input(&mut self, input: InputReplicant) {
        self.neuron_type.add_input(input)
    }

    pub fn inputs(&self) -> Option<&[InputReplicant]> {
        self.neuron_type.inputs()
    }

    pub fn is_output(&self) -> bool {
        self.neuron_type.is_output()
    }
    pub fn is_input(&self) -> bool {
        self.neuron_type.is_input()
    }
    pub fn is_hidden(&self) -> bool {
        self.neuron_type.is_hidden()
    }

    pub fn trim_inputs(&mut self, ids: Vec<Uuid>) {
        self.neuron_type.trim_inputs(ids)
    }

    pub fn to_neuron(
        &self,
        neurons: &mut Vec<Arc<RwLock<Neuron>>>,
        replicants: &[Arc<RwLock<NeuronReplicant>>],
    ) -> Arc<RwLock<Neuron>> {
        for neuron in neurons.iter() {
            if neuron.read().unwrap().id() == self.id() {
                return Arc::clone(neuron);
            }
        }

        let neuron_type = if let Some(inputs) = self.inputs() {
            let mut new_inputs = Vec::with_capacity(inputs.len());
            for input in inputs {
                if let Some(input_neuron) = input.neuron() {
                    let neuron = input_neuron.read().unwrap().to_neuron(neurons, replicants);
                    new_inputs.push(NeuronInput::new(neuron, input.weight()));
                }
            }

            if self.is_hidden() {
                NeuronType::Hidden {
                    inputs: new_inputs,
                    activation: self.activation().unwrap().as_fn(),
                    bias: self.bias().unwrap(),
                }
            } else {
                NeuronType::Output {
                    inputs: new_inputs,
                    activation: self.activation().unwrap().as_fn(),
                    bias: self.bias().unwrap(),
                }
            }
        } else {
            NeuronType::Input
        };

        let neuron = Neuron::new(self.id, neuron_type);

        Arc::new(RwLock::new(neuron))
    }
}
