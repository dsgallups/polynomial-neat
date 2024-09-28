use std::{
    collections::HashMap,
    fmt::Debug,
    hash::{BuildHasher, Hash},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
pub struct Variable<T> {
    pub var: T,
    pub exponent: i32,
}

impl<T> Variable<T> {
    pub fn new(var: T, exponent: i32) -> Self {
        Self { var, exponent }
    }

    pub fn exponent(&self) -> i32 {
        self.exponent
    }
    pub fn var(&self) -> &T {
        &self.var
    }
}

impl<T: Debug + Hash + Eq> Variable<T> {
    pub fn map_operands<V: Clone + Debug, S: BuildHasher>(
        self,
        operands: &HashMap<T, V, S>,
    ) -> Variable<V> {
        let Some(new_var) = operands.get(&self.var).cloned() else {
            panic!("couldn't find {:?}\noperands: {:#?}", self.var, operands);
        };

        Variable {
            var: new_var,
            exponent: self.exponent,
        }
    }
}
