/// Contains useful structs for serializing/deserializing a [`NeuronTopology`]
//#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
//#[cfg(feature = "serde")]
pub mod nnt_serde;

/// Contains structs and traits used for activation functions.
pub mod activation;

pub use activation::*;

use std::collections::HashSet;

use genetic_rs::prelude::*;
use rand::prelude::*;

use serde::{Deserialize, Serialize};

use crate::activation_fn;

/// A stateless neural network topology.
/// This is the struct you want to use in your agent's inheritance.
/// See [`NeuralNetwork::from`][crate::NeuralNetwork::from] for how to convert this to a runnable neural network.
#[derive(Debug, Clone)]
pub struct NeuralNetworkTopology<'f, const I: usize, const O: usize> {
    /// The input layer of the neural network. Uses a fixed length of `I`.
    pub input_layer: [NeuronTopology<'f>; I],

    /// The hidden layers of the neural network. Because neurons have a flexible connection system, all of them exist in the same flat vector.
    pub hidden_layers: Vec<NeuronTopology<'f>>,

    /// The output layer of the neural netowrk. Uses a fixed length of `O`.
    pub output_layer: [NeuronTopology<'f>; O],

    /// The mutation rate used in [`NeuralNetworkTopology::mutate`] after crossover/division.
    pub mutation_rate: f32,

    /// The number of mutation passes (and thus, maximum number of possible mutations that can occur for each entity in the generation).
    pub mutation_passes: usize,
}

impl<'f, const I: usize, const O: usize> NeuralNetworkTopology<'f, I, O> {
    /// Creates a new [`NeuralNetworkTopology`].
    pub fn new(mutation_rate: f32, mutation_passes: usize, rng: &mut impl Rng) -> Self {
        let input_layer: [NeuronTopology; I] = (0..I)
            .map(|_| {
                NeuronTopology::new_with_activation(vec![], activation_fn!(linear_activation), rng)
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let output_layer: [NeuronTopology; O] = (0..O)
            .map(|_| {
                // random number of connections to random input neurons.
                let input = (0..rng.gen_range(1..=I))
                    .map(|_| {
                        let mut already_chosen = Vec::new();
                        let mut i = rng.gen_range(0..I);
                        while already_chosen.contains(&i) {
                            i = rng.gen_range(0..I);
                        }

                        already_chosen.push(i);

                        NeuronLocation::Input(i)
                    })
                    .collect();

                NeuronTopology::new_with_activation(input, activation_fn!(sigmoid), rng)
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            input_layer,
            hidden_layers: vec![],
            output_layer,
            mutation_rate,
            mutation_passes,
        }
    }

    /// Creates a new connection between the neurons.
    /// If the connection is cyclic, it does not add a connection and returns false.
    /// Otherwise, it returns true.
    pub fn add_connection(
        &mut self,
        from: NeuronLocation,
        to: NeuronLocation,
        weight: f32,
    ) -> bool {
        if self.is_connection_cyclic(from, to) {
            return false;
        }

        // Add the connection since it is not cyclic
        self.get_neuron_mut(to).inputs.push((from, weight));

        true
    }

    fn is_connection_cyclic(&self, from: NeuronLocation, to: NeuronLocation) -> bool {
        if to.is_input() || from.is_output() {
            return true;
        }

        let mut visited = HashSet::new();
        self.dfs(from, to, &mut visited)
    }

    // TODO rayon implementation
    fn dfs(
        &self,
        current: NeuronLocation,
        target: NeuronLocation,
        visited: &mut HashSet<NeuronLocation>,
    ) -> bool {
        if current == target {
            return true;
        }

        visited.insert(current);

        let n = self.get_neuron(current);

        for &(input, _) in &n.inputs {
            if !visited.contains(&input) && self.dfs(input, target, visited) {
                return true;
            }
        }

        visited.remove(&current);
        false
    }

    /// Gets a neuron pointer from a [`NeuronLocation`].
    /// You shouldn't ever need to directly call this unless you are doing complex custom mutations.
    pub fn get_neuron(&self, loc: NeuronLocation) -> &NeuronTopology<'f> {
        match loc {
            NeuronLocation::Input(i) => &self.input_layer[i],
            NeuronLocation::Hidden(i) => &self.hidden_layers[i],
            NeuronLocation::Output(i) => &self.output_layer[i],
        }
    }

    /// Gets a neuron pointer from a [`NeuronLocation`].
    /// You shouldn't ever need to directly call this unless you are doing complex custom mutations.
    pub fn get_neuron_mut(&mut self, loc: NeuronLocation) -> &mut NeuronTopology<'f> {
        match loc {
            NeuronLocation::Input(i) => &mut self.input_layer[i],
            NeuronLocation::Hidden(i) => &mut self.hidden_layers[i],
            NeuronLocation::Output(i) => &mut self.output_layer[i],
        }
    }
    /// Gets a random neuron and its location.
    pub fn rand_neuron(&self, rng: &mut impl Rng) -> (&NeuronTopology<'f>, NeuronLocation) {
        match rng.gen_range(0..3) {
            0 => {
                let i = rng.gen_range(0..self.input_layer.len());
                (&self.input_layer[i], NeuronLocation::Input(i))
            }
            1 if !self.hidden_layers.is_empty() => {
                let i = rng.gen_range(0..self.hidden_layers.len());
                (&self.hidden_layers[i], NeuronLocation::Hidden(i))
            }
            _ => {
                let i = rng.gen_range(0..self.output_layer.len());
                (&self.output_layer[i], NeuronLocation::Output(i))
            }
        }
    }

    /// Gets a random neuron and its location.
    pub fn rand_neuron_mut(
        &mut self,
        rng: &mut impl Rng,
    ) -> (&mut NeuronTopology<'f>, NeuronLocation) {
        match rng.gen_range(0..3) {
            0 => {
                let i = rng.gen_range(0..self.input_layer.len());
                (&mut self.input_layer[i], NeuronLocation::Input(i))
            }
            1 if !self.hidden_layers.is_empty() => {
                let i = rng.gen_range(0..self.hidden_layers.len());
                (&mut self.hidden_layers[i], NeuronLocation::Hidden(i))
            }
            _ => {
                let i = rng.gen_range(0..self.output_layer.len());
                (&mut self.output_layer[i], NeuronLocation::Output(i))
            }
        }
    }

    fn delete_neuron(&mut self, loc: NeuronLocation) -> NeuronTopology<'f> {
        if !loc.is_hidden() {
            panic!("Invalid neuron deletion");
        }

        let index = loc.unwrap();
        let neuron = self.hidden_layers.remove(index);

        for nw in self.hidden_layers.iter_mut() {
            nw.inputs = nw
                .inputs
                .iter()
                .filter_map(|&(input_loc, w)| {
                    if !input_loc.is_hidden() {
                        return Some((input_loc, w));
                    }

                    if input_loc.unwrap() == index {
                        return None;
                    }

                    if input_loc.unwrap() > index {
                        return Some((NeuronLocation::Hidden(input_loc.unwrap() - 1), w));
                    }

                    Some((input_loc, w))
                })
                .collect();
        }

        for nw in self.output_layer.iter_mut() {
            nw.inputs = nw
                .inputs
                .iter()
                .filter_map(|&(input_loc, w)| {
                    if !input_loc.is_hidden() {
                        return Some((input_loc, w));
                    }

                    if input_loc.unwrap() == index {
                        return None;
                    }

                    if input_loc.unwrap() > index {
                        return Some((NeuronLocation::Hidden(input_loc.unwrap() - 1), w));
                    }

                    Some((input_loc, w))
                })
                .collect();
        }

        neuron
    }
}

/// todo:
/// - Get rid of these while loops
impl<'f, const I: usize, const O: usize> RandomlyMutable for NeuralNetworkTopology<'f, I, O> {
    fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
        for _ in 0..self.mutation_passes {
            if rng.gen::<f32>() <= rate {
                // split preexisting connection
                let ((loc, w), (rn_loc, i)) = {
                    let (mut n, mut rn_loc) = self.rand_neuron_mut(rng);

                    while n.inputs.is_empty() {
                        (n, rn_loc) = self.rand_neuron_mut(rng);
                    }

                    let i = rng.gen_range(0..n.inputs.len());
                    (n.inputs.remove(i), (rn_loc, i))
                };

                let loc3 = NeuronLocation::Hidden(self.hidden_layers.len());

                let n3 = NeuronTopology::new(vec![loc], ActivationScope::HIDDEN, rng);

                self.hidden_layers.push(n3);

                self.get_neuron_mut(rn_loc).inputs.insert(i, (loc3, w));
            }

            if rng.gen::<f32>() <= rate {
                // add a connection
                let (_, mut loc1) = self.rand_neuron(rng);
                let (_, mut loc2) = self.rand_neuron(rng);

                while loc1.is_output() || !self.add_connection(loc1, loc2, rng.gen::<f32>()) {
                    (_, loc1) = self.rand_neuron(rng);
                    (_, loc2) = self.rand_neuron(rng);
                }
            }

            if rng.gen::<f32>() <= rate && !self.hidden_layers.is_empty() {
                // remove a neuron
                let (_, mut loc) = self.rand_neuron(rng);

                while !loc.is_hidden() {
                    (_, loc) = self.rand_neuron(rng);
                }

                // delete the neuron
                self.delete_neuron(loc);
            }

            if rng.gen::<f32>() <= rate {
                // mutate a connection
                let (mut n, _) = self.rand_neuron_mut(rng);

                while n.inputs.is_empty() {
                    (n, _) = self.rand_neuron_mut(rng);
                }

                let i = rng.gen_range(0..n.inputs.len());
                let (_, w) = &mut n.inputs[i];
                *w += rng.gen_range(-1.0..1.0) * rate;
            }

            if rng.gen::<f32>() <= rate {
                // mutate bias
                let (n, _) = self.rand_neuron_mut(rng);

                n.bias += rng.gen_range(-1.0..1.0) * rate;
            }

            if rng.gen::<f32>() <= rate && !self.hidden_layers.is_empty() {
                // mutate activation function
                let reg = ACTIVATION_REGISTRY.read().unwrap();
                let activations = reg.activations_in_scope(ActivationScope::HIDDEN);

                let (mut n, mut loc) = self.rand_neuron_mut(rng);

                while !loc.is_hidden() {
                    (n, loc) = self.rand_neuron_mut(rng);
                }

                // should probably not clone, but its not a huge efficiency issue anyways
                n.activation = activations[rng.gen_range(0..activations.len())].clone();
            }
        }
    }
}

impl<'f, const I: usize, const O: usize> DivisionReproduction for NeuralNetworkTopology<'f, I, O> {
    fn divide(&self, rng: &mut impl rand::Rng) -> Self {
        let mut child = self.clone();
        child.mutate(self.mutation_rate, rng);
        child
    }
}

impl<'f, const I: usize, const O: usize> PartialEq for NeuralNetworkTopology<'f, I, O> {
    fn eq(&self, other: &Self) -> bool {
        if self.mutation_rate != other.mutation_rate
            || self.mutation_passes != other.mutation_passes
        {
            return false;
        }

        for i in 0..I {
            if self.input_layer[i] != other.input_layer[i] {
                return false;
            }
        }

        // TODO(dsgallups): this could be a bug
        for i in 0..self.hidden_layers.len().min(other.hidden_layers.len()) {
            if self.hidden_layers[i] != other.hidden_layers[i] {
                return false;
            }
        }

        for i in 0..O {
            if self.output_layer[i] != other.output_layer[i] {
                return false;
            }
        }

        true
    }
}

#[cfg(feature = "crossover")]
impl<'f, const I: usize, const O: usize> CrossoverReproduction for NeuralNetworkTopology<'f, I, O> {
    fn crossover(&self, other: &Self, rng: &mut impl rand::Rng) -> Self {
        let input_layer = self.input_layer.clone();

        let mut hidden_layers =
            Vec::with_capacity(self.hidden_layers.len().max(other.hidden_layers.len()));

        for i in 0..hidden_layers.len() {
            if rng.gen::<f32>() <= 0.5 {
                if let Some(n) = self.hidden_layers.get(i) {
                    let mut n = n.clone();
                    n.inputs
                        .retain(|(l, _)| input_exists(*l, &input_layer, &hidden_layers));
                    hidden_layers[i] = n;
                    continue;
                }
            }

            let mut n = other.hidden_layers[i].clone();

            n.inputs
                .retain(|(l, _)| input_exists(*l, &input_layer, &hidden_layers));
            hidden_layers[i] = n;
        }

        let mut output_layer = self.output_layer.clone();

        for (i, n) in self.output_layer.iter().enumerate() {
            if rng.gen::<f32>() <= 0.5 {
                let mut n = n.clone();
                n.inputs
                    .retain(|(l, _)| input_exists(*l, &input_layer, &hidden_layers));
                output_layer[i] = n;
                continue;
            }

            let mut n = other.output_layer[i].clone();

            n.inputs
                .retain(|(l, _)| input_exists(*l, &input_layer, &hidden_layers));
            output_layer[i] = n;
        }

        let mut child = Self {
            input_layer,
            hidden_layers,
            output_layer,
            mutation_rate: self.mutation_rate,
            mutation_passes: self.mutation_passes,
        };

        child.mutate(self.mutation_rate, rng);

        child
    }
}

#[cfg(feature = "crossover")]
fn input_exists<const I: usize>(
    loc: NeuronLocation,
    input: &[NeuronTopology<'_>; I],
    hidden: &[NeuronTopology<'_>],
) -> bool {
    match loc {
        NeuronLocation::Input(i) => i < input.len(),
        NeuronLocation::Hidden(i) => i < hidden.len(),
        NeuronLocation::Output(_) => false,
    }
}

/// A stateless version of [`Neuron`][crate::Neuron].
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct NeuronTopology<'f> {
    /// The input locations and weights.
    pub inputs: Vec<(NeuronLocation, f32)>,

    /// The neuron's bias.
    pub bias: f32,

    /// The neuron's activation function.
    pub activation: ActivationFn<'f>,
}

impl<'f> NeuronTopology<'f> {
    /// Creates a new neuron with the given input locations.
    pub fn new(
        inputs: Vec<NeuronLocation>,
        current_scope: ActivationScope,
        rng: &mut impl Rng,
    ) -> Self {
        let reg = ACTIVATION_REGISTRY.read().unwrap();
        let activations = reg.activations_in_scope(current_scope);

        Self::new_with_activations(inputs, activations, rng)
    }

    /// Takes a collection of activation functions and chooses a random one to use.
    pub fn new_with_activations(
        inputs: Vec<NeuronLocation>,
        activations: impl IntoIterator<Item = ActivationFn<'f>>,
        rng: &mut impl Rng,
    ) -> Self {
        let mut activations: Vec<_> = activations.into_iter().collect();

        Self::new_with_activation(
            inputs,
            activations.remove(rng.gen_range(0..activations.len())),
            rng,
        )
    }

    /// Creates a neuron with the activation.
    pub fn new_with_activation(
        inputs: Vec<NeuronLocation>,
        activation: ActivationFn<'f>,
        rng: &mut impl Rng,
    ) -> Self {
        let inputs = inputs
            .into_iter()
            .map(|i| (i, rng.gen_range(-1.0..1.0)))
            .collect();

        Self {
            inputs,
            bias: rng.gen(),
            activation,
        }
    }
}

/// A pseudo-pointer of sorts used to make structural conversions very fast and easy to write.
#[derive(Hash, Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NeuronLocation {
    /// Points to a neuron in the input layer at contained index.
    Input(usize),

    /// Points to a neuron in the hidden layer at contained index.
    Hidden(usize),

    /// Points to a neuron in the output layer at contained index.
    Output(usize),
}

impl NeuronLocation {
    /// Returns `true` if it points to the input layer. Otherwise, returns `false`.
    pub fn is_input(&self) -> bool {
        matches!(self, Self::Input(_))
    }

    /// Returns `true` if it points to the hidden layer. Otherwise, returns `false`.
    pub fn is_hidden(&self) -> bool {
        matches!(self, Self::Hidden(_))
    }

    /// Returns `true` if it points to the output layer. Otherwise, returns `false`.
    pub fn is_output(&self) -> bool {
        matches!(self, Self::Output(_))
    }

    /// Retrieves the index value, regardless of layer. Does not consume.
    pub fn unwrap(&self) -> usize {
        match self {
            Self::Input(i) => *i,
            Self::Hidden(i) => *i,
            Self::Output(i) => *i,
        }
    }
}
