#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolyComponent {
    exponent: i32,
    weight: f32,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Polynomial {
    ops: Vec<PolyComponent>,
}

impl Polynomial {
    pub fn with_operation(mut self, exponent: i32, weight: f32) -> Self {
        match self.ops.iter_mut().find(|op| op.exponent == exponent) {
            Some(op) => {
                op.weight += weight;
            }
            None => self.ops.push(PolyComponent { exponent, weight }),
        }
        self
    }
    pub fn handle_operation(&mut self, exponent: i32, weight: f32) -> &mut Self {
        match self.ops.iter_mut().find(|op| op.exponent == exponent) {
            Some(op) => {
                op.weight += weight;
            }
            None => self.ops.push(PolyComponent { exponent, weight }),
        }
        self
    }

    pub fn sort_by_exponent(&mut self) {
        self.ops.sort_by(|a, b| a.exponent.cmp(&b.exponent));
    }

    pub fn components(&self) -> &[PolyComponent] {
        &self.ops
    }

    pub fn expand(&mut self, other: &Polynomial, exponent: i32, weight: f32) -> &mut Self {
        todo!()
    }
}
