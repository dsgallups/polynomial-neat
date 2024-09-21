use rand::Rng;

#[derive(Clone, Debug)]
pub enum MutationAction {
    SplitConnection,
    AddConnection,
    RemoveNeuron,
    MutateWeight,
    MutateBias,
    MutateActivationFunction,
}

pub(crate) trait MutationRateExt {
    fn gen_rate(&mut self) -> u8;

    fn gen_mutation_action(&mut self, chances: &MutationChances) -> MutationAction;
}

impl<T: Rng> MutationRateExt for T {
    fn gen_rate(&mut self) -> u8 {
        self.gen_range(0..=100)
    }

    fn gen_mutation_action(&mut self, chances: &MutationChances) -> MutationAction {
        use MutationAction::*;

        let rate = self.gen_rate() as f32;

        // note that mutation chance values add up to 100.

        if rate <= chances.split_connection() {
            SplitConnection
        } else if rate <= chances.split_connection() + chances.add_connection() {
            AddConnection
        } else if rate
            <= chances.split_connection() + chances.add_connection() + chances.remove_connection()
        {
            RemoveNeuron
        } else if rate
            <= chances.split_connection()
                + chances.add_connection()
                + chances.remove_connection()
                + chances.mutate_weight()
        {
            MutateWeight
        } else if rate
            <= chances.split_connection()
                + chances.add_connection()
                + chances.remove_connection()
                + chances.mutate_weight()
                + chances.mutate_bias()
        {
            MutateBias
        } else {
            MutateActivationFunction
        }
    }
}

pub const MAX_MUTATIONS: u8 = 200;

#[derive(Clone, Copy, Debug)]
pub struct MutationChances {
    self_mutation: u8,
    split_connection: f32,
    add_connection: f32,
    remove_connection: f32,
    mutate_weight: f32,
    mutate_bias: f32,
    mutate_activation_fn: f32,
}

impl MutationChances {
    pub fn new(self_mutation_rate: u8) -> Self {
        let value = 100. / 6.;

        Self {
            self_mutation: self_mutation_rate,
            remove_connection: value,
            mutate_bias: value,
            split_connection: value,
            add_connection: value,
            mutate_activation_fn: value,
            mutate_weight: value,
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn new_from_raw(
        self_mutation: u8,
        split_connection: f32,
        add_connection: f32,
        remove_connection: f32,
        mutate_weight: f32,
        mutate_bias: f32,
        mutate_activation_fn: f32,
    ) -> Self {
        Self {
            self_mutation,
            split_connection,
            add_connection,
            remove_connection,
            mutate_weight,
            mutate_bias,
            mutate_activation_fn,
        }
    }

    pub fn adjust_mutation_chances(&mut self, rng: &mut impl Rng) {
        use MutationAction::*;
        const MAX_LOOP: u8 = 5;
        let mut loop_count = 0;
        while rng.gen_rate() < self.self_mutation() && loop_count < MAX_LOOP {
            let action = match rng.gen_range(0..6) {
                0 => SplitConnection,
                1 => AddConnection,
                2 => RemoveNeuron,
                3 => MutateWeight,
                4 => MutateBias,
                _ => MutateActivationFunction,
            };

            // Generate a random number between 1.0 and 10.0
            let value = rng.gen_range(0.0..=5.0);

            let add_to = if rng.gen_bool(0.5) { -value } else { value };

            match action {
                MutationAction::SplitConnection => {
                    self.adjust_split_connection(add_to);
                }
                MutationAction::AddConnection => {
                    self.adjust_add_connection(add_to);
                }
                MutationAction::RemoveNeuron => {
                    self.adjust_remove_connection(add_to);
                }
                MutationAction::MutateWeight => {
                    self.adjust_mutate_weight(add_to);
                }
                MutationAction::MutateBias => {
                    self.adjust_mutate_bias(add_to);
                }
                MutationAction::MutateActivationFunction => {
                    self.adjust_mutate_activation_fn(add_to);
                }
            }

            loop_count += 1;
        }

        self.adjust_self_mutation(rng);
    }

    pub fn self_mutation(&self) -> u8 {
        self.self_mutation
    }

    fn adjust_self_mutation(&mut self, rng: &mut impl Rng) {
        let rate: i8 = rng.gen_range(-1..=1);

        if rate < 0 && self.self_mutation == 0 {
            return;
        }

        if rate > 0 && self.self_mutation == 100 {
            return;
        }

        if rate.saturating_add(self.self_mutation as i8) < 0 {
            self.self_mutation = 0;
            return;
        }

        if rate.saturating_add(self.self_mutation as i8) > 100 {
            self.self_mutation = 100;
            return;
        }

        self.self_mutation = (self.self_mutation as i8 + rate) as u8;
    }

    pub fn split_connection(&self) -> f32 {
        self.split_connection
    }

    pub fn add_connection(&self) -> f32 {
        self.add_connection
    }

    pub fn remove_connection(&self) -> f32 {
        self.remove_connection
    }

    pub fn mutate_weight(&self) -> f32 {
        self.mutate_weight
    }

    pub fn mutate_bias(&self) -> f32 {
        self.mutate_bias
    }

    pub fn mutate_activation_fn(&self) -> f32 {
        self.mutate_activation_fn
    }

    fn adjust_split_connection(&mut self, amt: f32) {
        self.split_connection += amt;

        if self.split_connection < 0. {
            self.split_connection = 0.;
        }

        self.recalculate();
    }

    fn adjust_add_connection(&mut self, amt: f32) {
        self.add_connection += amt;

        if self.add_connection < 0. {
            self.add_connection = 0.;
        }

        self.recalculate();
    }

    fn adjust_remove_connection(&mut self, amt: f32) {
        self.remove_connection += amt;

        if self.remove_connection < 0. {
            self.remove_connection = 0.;
        }

        self.recalculate();
    }

    fn adjust_mutate_weight(&mut self, amt: f32) {
        self.mutate_weight += amt;

        if self.mutate_weight < 0. {
            self.mutate_weight = 0.;
        }

        self.recalculate();
    }

    fn adjust_mutate_bias(&mut self, amt: f32) {
        self.mutate_bias += amt;

        if self.mutate_bias < 0. {
            self.mutate_bias = 0.;
        }

        self.recalculate();
    }

    fn adjust_mutate_activation_fn(&mut self, amt: f32) {
        self.mutate_activation_fn += amt;

        if self.mutate_activation_fn < 0. {
            self.mutate_activation_fn = 0.;
        }

        self.recalculate();
    }

    fn recalculate(&mut self) {
        let total = self.split_connection
            + self.add_connection
            + self.remove_connection
            + self.mutate_weight
            + self.mutate_bias
            + self.mutate_activation_fn;

        self.split_connection = (self.split_connection * 100.) / total;
        self.add_connection = (self.add_connection * 100.) / total;
        self.remove_connection = (self.remove_connection * 100.) / total;
        self.mutate_weight = (self.mutate_weight * 100.) / total;
        self.mutate_bias = (self.mutate_bias * 100.) / total;
        self.mutate_activation_fn = (self.mutate_activation_fn * 100.) / total;
    }

    pub fn gen_mutation_actions(&self, rng: &mut impl Rng) -> Vec<MutationAction> {
        let mut actions = Vec::with_capacity(MAX_MUTATIONS as usize);

        let mut loop_count = 0;
        while rng.gen_rate() < self.self_mutation() && loop_count < MAX_MUTATIONS {
            actions.push(rng.gen_mutation_action(self));
            loop_count += 1;
        }

        actions
    }
}

#[test]
pub fn adjust_mutation_chances() {
    let mut chances = MutationChances::new(50);

    chances.adjust_split_connection(10.);

    chances.adjust_mutate_activation_fn(-10.);

    chances.adjust_add_connection(-10.);

    chances.adjust_remove_connection(10.);

    chances.adjust_mutate_weight(-10.);

    let total = chances.split_connection
        + chances.add_connection
        + chances.remove_connection
        + chances.mutate_weight
        + chances.mutate_bias
        + chances.mutate_activation_fn;
    let diff = (100. - total).abs();

    assert!(diff <= 0.0001);
}

#[test]
pub fn check_mutate() {
    let mut rng = rand::thread_rng();

    let mut chances = MutationChances::new(50);

    for _ in 0..100 {
        chances.adjust_mutation_chances(&mut rng);

        println!("{:?}", chances);

        let total = chances.split_connection
            + chances.add_connection
            + chances.remove_connection
            + chances.mutate_weight
            + chances.mutate_bias
            + chances.mutate_activation_fn;

        let diff = (100. - total).abs();

        assert!(diff <= 0.0001);
    }
}
