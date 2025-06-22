use std::sync::{Arc, RwLock};

use rayon::iter::{IndexedParallelIterator as _, IntoParallelRefIterator, ParallelIterator as _};

use crate::prelude::*;

/// A simple CPU-based polynomial neural network for inference.
///
/// This struct represents an executable neural network that can perform
/// forward passes on input data. It maintains references to all neurons
/// in the network organized into layers for efficient computation.
///
/// # Architecture
///
/// The network consists of:
/// - **Input layer**: Neurons that receive external inputs
/// - **Hidden neurons**: Neurons that process intermediate values
/// - **Output layer**: Neurons that produce the final outputs
///
/// All neurons are stored in a single vector with the input and output
/// layers maintaining references to their respective neurons.
///
/// # Thread Safety
///
/// The network uses `Arc<RwLock<>>` for thread-safe access to neurons,
/// allowing parallel evaluation using Rayon.
///
/// # Example
///
/// ```rust
/// use polynomial_neat::prelude::*;
/// use polynomial_neat::topology::mutation::MutationChances;
///
/// // Create a topology
/// let mutations = MutationChances::new(50);
/// let topology = PolyNetworkTopology::new(2, 1, mutations, &mut rand::rng());
///
/// // Convert to executable network
/// let network = topology.to_simple_network();
///
/// // Run inference
/// let inputs = vec![1.0, 0.5];
/// let outputs: Vec<f32> = network.predict(&inputs).collect();
/// println!("Network output: {:?}", outputs);
/// ```
pub struct SimplePolyNetwork {
    // contains all neurons
    neurons: Vec<Arc<RwLock<SimpleNeuron>>>,
    // contains the input neurons. cloned arc of neurons in neurons
    input_layer: Vec<Arc<RwLock<SimpleNeuron>>>,
    // contains the output neurons. cloned arc of neurons in neurons
    output_layer: Vec<Arc<RwLock<SimpleNeuron>>>,
}

impl SimplePolyNetwork {
    /// Perform a forward pass through the network with the given inputs.
    ///
    /// This method:
    /// 1. Resets all neuron states to prepare for a fresh computation
    /// 2. Sets the input values on input neurons
    /// 3. Propagates values through the network
    /// 4. Returns the outputs from output neurons
    ///
    /// # Arguments
    /// * `inputs` - Slice of input values. Length should match the number of input neurons.
    ///
    /// # Returns
    /// An iterator over the output values from the network's output neurons.
    ///
    /// # Example
    /// ```rust
    /// # use polynomial_neat::prelude::*;
    /// # use polynomial_neat::topology::mutation::MutationChances;
    /// # let mutations = MutationChances::new(50);
    /// # let topology = PolyNetworkTopology::new(2, 1, mutations, &mut rand::rng());
    /// # let network = topology.to_simple_network();
    /// // Predict with two inputs
    /// let outputs: Vec<f32> = network.predict(&[1.0, 0.5]).collect();
    /// assert_eq!(outputs.len(), 1); // One output neuron
    /// ```
    ///
    /// # Note
    /// If there are more inputs than input neurons, extra inputs are ignored.
    /// If there are fewer inputs than input neurons, the remaining neurons
    /// will have their state set to 0.
    pub fn predict(&self, inputs: &[f32]) -> impl Iterator<Item = f32> {
        // reset all states first
        self.neurons.par_iter().for_each(|neuron| {
            let mut neuron = neuron.write().unwrap();
            neuron.flush_state();
        });
        inputs.par_iter().enumerate().for_each(|(index, value)| {
            let Some(nw) = self.input_layer.get(index) else {
                //sim
                return;
                //panic!("couldn't flush i {}", index);
            };
            let mut nw = nw.write().unwrap();
            nw.override_state(*value);
        });

        let outputs = self
            .output_layer
            .par_iter()
            .fold(Vec::new, |mut values, neuron| {
                let mut neuron = neuron.write().unwrap();

                values.push(neuron.activate());

                values
            })
            .collect_vec_list();

        outputs
            .into_iter()
            .flat_map(|outer_vec| outer_vec.into_iter())
            .flat_map(|inner_vec| inner_vec.into_iter())
    }

    /// Create a network from raw components.
    ///
    /// This is a low-level constructor that assumes the provided components
    /// are correctly structured. The input and output layer vectors should
    /// contain references to neurons that also exist in the main neurons vector.
    ///
    /// # Arguments
    /// * `neurons` - All neurons in the network
    /// * `input_layer` - References to input neurons
    /// * `output_layer` - References to output neurons
    ///
    /// # Example
    /// ```rust
    /// # use polynomial_neat::prelude::*;
    /// # use std::sync::{Arc, RwLock};
    /// # use uuid::Uuid;
    /// // Create neurons manually
    /// let input = Arc::new(RwLock::new(SimpleNeuron::new(Uuid::new_v4(), None)));
    /// let output = Arc::new(RwLock::new(SimpleNeuron::new(Uuid::new_v4(), None)));
    ///
    /// let neurons = vec![input.clone(), output.clone()];
    /// let input_layer = vec![input];
    /// let output_layer = vec![output];
    ///
    /// let network = SimplePolyNetwork::from_raw_parts(neurons, input_layer, output_layer);
    /// ```
    pub fn from_raw_parts(
        neurons: Vec<Arc<RwLock<SimpleNeuron>>>,
        input_layer: Vec<Arc<RwLock<SimpleNeuron>>>,
        output_layer: Vec<Arc<RwLock<SimpleNeuron>>>,
    ) -> Self {
        Self {
            neurons,
            input_layer,
            output_layer,
        }
    }

    /// Generate a human-readable summary of the network's structure.
    ///
    /// # Returns
    /// A formatted string describing the network's neuron counts.
    ///
    /// # Example
    /// ```rust
    /// # use polynomial_neat::prelude::*;
    /// # use polynomial_neat::topology::mutation::MutationChances;
    /// # let mutations = MutationChances::new(50);
    /// # let topology = PolyNetworkTopology::new(2, 1, mutations, &mut rand::rng());
    /// # let network = topology.to_simple_network();
    /// println!("{}", network.summarize());
    /// // Output: Network with
    /// // 3 total nodes
    /// // 2 input nodes
    /// // 1 output nodes
    /// ```
    pub fn summarize(&self) -> String {
        format!(
            "Network with \n{} total nodes\n{} input nodes\n{} output nodes",
            self.num_nodes(),
            self.num_inputs(),
            self.num_outputs()
        )
    }

    /// Get the total number of neurons in the network.
    ///
    /// This includes input, hidden, and output neurons.
    pub fn num_nodes(&self) -> usize {
        self.neurons.len()
    }

    /// Get the number of input neurons.
    ///
    /// This determines how many input values the network expects.
    pub fn num_inputs(&self) -> usize {
        self.input_layer.len()
    }

    /// Get the number of output neurons.
    ///
    /// This determines how many output values the network produces.
    pub fn num_outputs(&self) -> usize {
        self.output_layer.len()
    }

    /// Generate a detailed debug representation of the network structure.
    ///
    /// This method provides a comprehensive view of:
    /// - All neurons with their IDs and types
    /// - Connection patterns between neurons
    /// - Layer organization
    ///
    /// The output format shows neuron indices in parentheses and connection
    /// targets in square brackets.
    ///
    /// # Returns
    /// A formatted string with detailed network structure information.
    ///
    /// # Example Output
    /// ```text
    /// neurons:
    /// ((0) a1b2c3[input]: N/A)
    /// ((1) d4e5f6[hidden]: [(0)])
    /// ((2) g7h8i9[output]: [(1)])
    ///
    /// input_layer:
    /// ((0) a1b2c3[input]: N/A)
    ///
    /// output layer:
    /// ((0) g7h8i9[output]: [(1)])
    /// ```
    pub fn debug_str(&self) -> String {
        let mut str = "neurons: \n".to_string();
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
                        let n = input.input().read().unwrap();

                        let loc = self
                            .neurons
                            .iter()
                            .position(|neuron| neuron.read().unwrap().id() == n.id())
                            .unwrap();

                        str.push_str(&format!("({})", loc));
                    }
                    str.push(']')
                }

                None => {
                    str.push_str("N/A");
                }
            }

            str.push(')');
        }

        str.push_str("\n\ninput_layer:");

        for (neuron_index, neuron) in self.input_layer.iter().enumerate() {
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
                        let n = input.input().read().unwrap();

                        let loc = self
                            .neurons
                            .iter()
                            .position(|neuron| neuron.read().unwrap().id() == n.id())
                            .unwrap();

                        str.push_str(&format!("({})", loc));
                    }
                    str.push(']')
                }

                None => {
                    str.push_str("N/A");
                }
            }

            str.push(')');
        }

        str.push_str("\n\noutput layer:");

        for (neuron_index, neuron) in self.output_layer.iter().enumerate() {
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
                        let n = input.input().read().unwrap();

                        let loc = self
                            .neurons
                            .iter()
                            .position(|neuron| neuron.read().unwrap().id() == n.id())
                            .unwrap();

                        str.push_str(&format!("({})", loc));
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

    /// Create an executable network from a topology representation.
    ///
    /// This method converts a `PolyNetworkTopology` (which represents the
    /// structure and evolution parameters) into a `SimplePolyNetwork` that
    /// can perform inference.
    ///
    /// The conversion process:
    /// 1. Creates `SimpleNeuron` instances from topology neurons
    /// 2. Establishes connections between neurons
    /// 3. Organizes neurons into input and output layers
    ///
    /// # Arguments
    /// * `topology` - The network topology to convert
    ///
    /// # Returns
    /// A new `SimplePolyNetwork` ready for inference
    ///
    /// # Example
    /// ```rust
    /// # use polynomial_neat::prelude::*;
    /// # use polynomial_neat::topology::mutation::MutationChances;
    /// let mutations = MutationChances::new(50);
    /// let topology = PolyNetworkTopology::new(3, 2, mutations, &mut rand::rng());
    ///
    /// // Convert to executable network
    /// let network = SimplePolyNetwork::from_topology(&topology);
    ///
    /// // Now ready for inference
    /// let outputs: Vec<f32> = network.predict(&[1.0, 2.0, 3.0]).collect();
    /// ```
    pub fn from_topology(topology: &PolyNetworkTopology) -> Self {
        let mut neurons: Vec<Arc<RwLock<SimpleNeuron>>> =
            Vec::with_capacity(topology.neurons().len());
        let mut input_layer: Vec<Arc<RwLock<SimpleNeuron>>> = Vec::new();
        let mut output_layer: Vec<Arc<RwLock<SimpleNeuron>>> = Vec::new();

        for neuron_replicant in topology.neurons() {
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

        SimplePolyNetwork::from_raw_parts(neurons, input_layer, output_layer)
    }
}
