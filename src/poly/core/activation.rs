use rand::Rng;

pub struct Bias;

impl Bias {
    pub fn rand(rng: &mut impl Rng) -> f32 {
        rng.random()
    }
}

pub struct Exponent;
impl Exponent {
    pub fn rand(rng: &mut impl Rng) -> i32 {
        rng.random_range(0..=1)
    }
}
