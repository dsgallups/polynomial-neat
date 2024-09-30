use rand::Rng;

/// Defines a weight and a reference to an input
#[derive(Clone, Debug)]
pub struct PolyInput<I> {
    input: I,
    weight: f32,
    exp: i32,
}

impl<I> PolyInput<I> {
    pub fn new(input: I, weight: f32, exp: i32) -> Self {
        Self { input, weight, exp }
    }

    pub fn new_rand(input: I, rng: &mut impl Rng) -> Self {
        Self {
            input,
            weight: rng.gen_range(-1.0..=1.0),
            exp: rng.gen_range(0..=2),
        }
    }

    pub fn input(&self) -> &I {
        &self.input
    }

    pub fn weight(&self) -> f32 {
        self.weight
    }

    pub fn adjust_weight(&mut self, by: f32) {
        self.weight += by;
    }

    pub fn exponent(&self) -> i32 {
        self.exp
    }

    pub fn adjust_exp(&mut self, by: i32) {
        self.exp += by;
    }
}
