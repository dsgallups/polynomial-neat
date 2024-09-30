use rand::Rng;

pub struct Bias;

impl Bias {
    pub fn rand(rng: &mut impl Rng) -> f32 {
        rng.gen()
    }
}

pub struct Exponent;
impl Exponent {
    pub fn rand(rng: &mut impl Rng) -> i32 {
        rng.gen_range(0..=1)
    }
}
