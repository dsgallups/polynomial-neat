use serde::{Deserialize, Deserializer, Serialize, Serializer};

use bitflags::bitflags;
use std::{
    collections::HashMap,
    fmt,
    sync::{LazyLock, RwLock},
};

use crate::prelude::*;

/// Creates an [`ActivationFn`] object from a function
#[macro_export]
macro_rules! activation_fn {
    ($F: path) => {
        ActivationFn::new(&$F, ActivationScope::default(), stringify!($F).into())
    };

    ($F: path, $S: expr) => {
        ActivationFn::new(&$F, $S, stringify!($F).into())
    };

    {$($F: path),*} => {
        [$(activation_fn!($F)),*]
    };

    {$($F: path => $S: expr),*} => {
        [$(activation_fn!($F, $S)),*]
    }
}

pub static ACTIVATION_REGISTRY: LazyLock<RwLock<ActivationRegistry>> =
    LazyLock::new(|| RwLock::new(ActivationRegistry::default()));

//pub static ACTIVATION_REGISTRY: LazyLock<Arc<RwLock<ActivationRegistry>> = LazyLock::new();

/// Register an activation function to the registry.
pub fn register_activation(act: ActivationFn<'static>) {
    let mut reg = ACTIVATION_REGISTRY.write().unwrap();
    reg.register(act);
}

/// Registers multiple activation functions to the registry at once.
pub fn batch_register_activation(acts: impl IntoIterator<Item = ActivationFn<'static>>) {
    let mut reg = ACTIVATION_REGISTRY.write().unwrap();
    reg.batch_register(acts);
}

/// A registry of the different possible activation functions.
pub struct ActivationRegistry<'f> {
    /// The currently-registered activation functions.
    pub fns: HashMap<&'f str, ActivationFn<'f>>,
}

impl<'f> ActivationRegistry<'f> {
    /// Registers an activation function.
    pub fn register(&mut self, activation: ActivationFn<'f>) {
        self.fns.insert(activation.name, activation);
    }

    /// Registers multiple activation functions at once.
    pub fn batch_register(&mut self, activations: impl IntoIterator<Item = ActivationFn<'f>>) {
        for act in activations {
            self.register(act);
        }
    }

    /// Gets a Vec of all the activation functions registered. Unless you need an owned value, use [fns][ActivationRegistry::fns].values() instead.
    pub fn activations(&self) -> Vec<ActivationFn<'f>> {
        self.fns.values().cloned().collect()
    }

    /// Gets all activation functions that are valid for a scope.
    pub fn activations_in_scope(&self, scope: ActivationScope) -> Vec<ActivationFn<'f>> {
        let acts = self.activations();

        acts.into_iter()
            .filter(|a| a.scope != ActivationScope::NONE && a.scope.contains(scope))
            .collect()
    }
}

impl<'f> Default for ActivationRegistry<'f> {
    fn default() -> Self {
        let mut s = Self {
            fns: HashMap::new(),
        };

        s.batch_register(activation_fn! {
            sigmoid => ActivationScope::HIDDEN | ActivationScope::OUTPUT,
            relu => ActivationScope::HIDDEN | ActivationScope::OUTPUT,
            linear_activation => ActivationScope::INPUT | ActivationScope::HIDDEN | ActivationScope::OUTPUT,
            f32::tanh => ActivationScope::HIDDEN | ActivationScope::OUTPUT
        });

        s
    }
}

bitflags! {
    /// Specifies where an activation function can occur
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct ActivationScope: u8 {
        /// Whether the activation can be applied to the input layer.
        const INPUT = 0b001;

        /// Whether the activation can be applied to the hidden layer.
        const HIDDEN = 0b010;

        /// Whether the activation can be applied to the output layer.
        const OUTPUT = 0b100;

        /// The activation function will not be randomly placed anywhere
        const NONE = 0b000;
    }
}

impl Default for ActivationScope {
    fn default() -> Self {
        Self::HIDDEN
    }
}

impl From<&NeuronLocation> for ActivationScope {
    fn from(value: &NeuronLocation) -> Self {
        match value {
            NeuronLocation::Input(_) => Self::INPUT,
            NeuronLocation::Hidden(_) => Self::HIDDEN,
            NeuronLocation::Output(_) => Self::OUTPUT,
        }
    }
}

/// A trait that represents an activation method.
pub trait Activation: Sync {
    /// The activation function.
    fn activate(&self, n: f32) -> f32;
}

impl<F> Activation for F
where
    F: Fn(f32) -> f32 + Sync,
{
    fn activate(&self, n: f32) -> f32 {
        (self)(n)
    }
}

/// An activation function object that implements [`fmt::Debug`] and is [`Send`]
#[derive(Clone)]
pub struct ActivationFn<'f> {
    /// The actual activation function.
    pub func: &'f dyn Activation,

    /// The scope defining where the activation function can appear.
    pub scope: ActivationScope,
    pub(crate) name: &'f str,
}

impl<'f> ActivationFn<'f> {
    /// Creates a new ActivationFn object.
    pub fn new<F>(func: &'f F, scope: ActivationScope, name: &'f str) -> Self
    where
        F: Activation,
    {
        Self { func, name, scope }
    }
}

impl<'f> fmt::Debug for ActivationFn<'f> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.name)
    }
}

impl<'f> PartialEq for ActivationFn<'f> {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl<'f> Serialize for ActivationFn<'f> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name)
    }
}

impl<'f, 'a> Deserialize<'a> for ActivationFn<'f> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        let name = String::deserialize(deserializer)?;

        let reg = ACTIVATION_REGISTRY.read().unwrap();

        let f = reg.fns.get(name.as_str());

        if f.is_none() {
            panic!("Activation function {name} not found");
        }

        Ok(f.unwrap().clone())
    }
}

/// The sigmoid activation function.
pub fn sigmoid(n: f32) -> f32 {
    1. / (1. + std::f32::consts::E.powf(-n))
}

/// The ReLU activation function.
pub fn relu(n: f32) -> f32 {
    n.max(0.)
}

/// Activation function that does nothing.
pub fn linear_activation(n: f32) -> f32 {
    n
}
