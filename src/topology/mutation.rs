use rand::Rng;

/// Represents the different types of mutations that can occur during network evolution.
///
/// Each mutation type modifies the network topology or parameters in a specific way
/// to explore the solution space.
#[derive(Clone, Debug)]
pub enum MutationAction {
    /// Split an existing connection by adding a new neuron in between.
    /// This increases network complexity by adding a hidden layer neuron.
    SplitConnection,
    /// Add a new connection between two existing neurons.
    /// This creates new pathways for information flow.
    AddConnection,
    /// Remove a neuron and all its connections from the network.
    /// This simplifies the network by removing unnecessary complexity.
    RemoveNeuron,
    /// Modify the weight of an existing connection.
    /// This fine-tunes the strength of connections.
    MutateWeight,
    /// Modify the exponent of a polynomial activation.
    /// This changes the shape of the activation function.
    MutateExponent,
}

/// Extension trait for random number generators to generate mutation-related values.
pub(crate) trait MutationRateExt {
    /// Generate a random rate value between 0 and 100.
    fn gen_rate(&mut self) -> u8;

    /// Generate a random mutation action based on the configured chances.
    fn gen_mutation_action(&mut self, chances: &MutationChances) -> MutationAction;
}

impl<T: Rng> MutationRateExt for T {
    fn gen_rate(&mut self) -> u8 {
        self.random_range(0..=100)
    }

    fn gen_mutation_action(&mut self, chances: &MutationChances) -> MutationAction {
        use MutationAction::*;

        let rate = self.gen_rate() as f32;

        // note that mutation chance values add up to 100.

        // Handle edge case: when rate is 0, only select an action if its chance > 0
        if rate == 0.0 {
            if chances.split_connection() > 0.0 {
                return SplitConnection;
            } else if chances.add_connection() > 0.0 {
                return AddConnection;
            } else if chances.remove_connection() > 0.0 {
                return RemoveNeuron;
            } else if chances.mutate_weight() > 0.0 {
                return MutateWeight;
            } else {
                return MutateExponent;
            }
        }

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
        } else {
            MutateExponent
        }
    }
}

/// Maximum number of mutations that can occur in a single evolution step.
///
/// This prevents infinite mutation loops and ensures evolution remains tractable.
pub const MAX_MUTATIONS: u8 = 200;

/// Configuration for controlling mutation probabilities during network evolution.
///
/// This struct defines the likelihood of each type of mutation occurring when
/// a network evolves. All mutation chances are normalized to sum to 100%.
///
/// # Example
///
/// ```rust
/// use polynomial_neat::prelude::*;
/// use polynomial_neat::topology::mutation::MutationChances;
///
/// // Create balanced mutation chances (20% each)
/// let balanced = MutationChances::new(50);
///
/// // Create custom mutation chances
/// let custom = MutationChances::new_from_raw(
///     3,      // max mutations per evolution
///     80.0,   // 80% chance to split connections (add neurons)
///     50.0,   // relative chance to add connections
///     5.0,    // low chance to remove neurons
///     60.0,   // moderate chance to mutate weights
///     20.0    // low chance to mutate exponents
/// );
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MutationChances {
    /// Probability (0-100) of performing any mutation at all
    self_mutation: u8,
    /// Relative probability of splitting a connection
    split_connection: f32,
    /// Relative probability of adding a new connection
    add_connection: f32,
    /// Relative probability of removing a connection
    remove_connection: f32,
    /// Relative probability of mutating a weight
    mutate_weight: f32,
    /// Relative probability of mutating an exponent
    mutate_exponent: f32,
}

impl MutationChances {
    /// Create mutation chances with equal probability for each mutation type.
    ///
    /// # Arguments
    /// * `self_mutation_rate` - The probability (0-100) of performing mutations
    ///
    /// # Example
    /// ```rust
    /// # use polynomial_neat::topology::mutation::MutationChances;
    /// // 50% chance of mutation, with equal chances for each type
    /// let chances = MutationChances::new(50);
    /// ```
    pub fn new(self_mutation_rate: u8) -> Self {
        let value = 100. / 5.;

        Self {
            self_mutation: self_mutation_rate,
            remove_connection: value,
            mutate_exponent: value,
            split_connection: value,
            add_connection: value,
            mutate_weight: value,
        }
    }

    /// Create mutation chances that disable all mutations.
    ///
    /// Useful for testing or when you want to freeze network evolution.
    pub fn none() -> Self {
        Self {
            self_mutation: 0,
            split_connection: 0.,
            add_connection: 0.,
            remove_connection: 0.,
            mutate_weight: 0.,
            mutate_exponent: 0.,
        }
    }

    /// Create mutation chances with custom probabilities for each mutation type.
    ///
    /// The individual mutation chances will be automatically normalized to sum to 100%.
    ///
    /// # Arguments
    /// * `self_mutation` - Overall mutation probability (0-100)
    /// * `split_connection` - Relative chance of splitting connections
    /// * `add_connection` - Relative chance of adding connections
    /// * `remove_connection` - Relative chance of removing connections
    /// * `mutate_weight` - Relative chance of mutating weights
    /// * `mutate_exponent` - Relative chance of mutating exponents
    ///
    /// # Example
    /// ```rust
    /// # use polynomial_neat::topology::mutation::MutationChances;
    /// let chances = MutationChances::new_from_raw(
    ///     75,    // 75% chance of mutation
    ///     40.0,  // High chance for adding neurons
    ///     30.0,  // Moderate chance for new connections
    ///     5.0,   // Low chance for removing neurons
    ///     20.0,  // Moderate chance for weight changes
    ///     5.0    // Low chance for exponent changes
    /// );
    /// ```
    #[allow(clippy::type_complexity)]
    pub fn new_from_raw(
        self_mutation: u8,
        split_connection: f32,
        add_connection: f32,
        remove_connection: f32,
        mutate_weight: f32,
        mutate_exponent: f32,
    ) -> Self {
        let mut new = Self {
            self_mutation,
            split_connection,
            add_connection,
            remove_connection,
            mutate_weight,
            mutate_exponent,
        };
        new.recalculate();
        new
    }

    /// Randomly adjust the mutation chances themselves.
    ///
    /// This implements meta-evolution where the mutation parameters can evolve
    /// alongside the network topology. This allows the algorithm to adapt its
    /// exploration strategy over time.
    ///
    /// # Arguments
    /// * `rng` - Random number generator for probabilistic adjustments
    pub fn adjust_mutation_chances(&mut self, rng: &mut impl Rng) {
        use MutationAction::*;
        const MAX_LOOP: u8 = 5;
        let mut loop_count = 0;
        while rng.gen_rate() < self.self_mutation() && loop_count < MAX_LOOP {
            let action = match rng.random_range(0..5) {
                0 => SplitConnection,
                1 => AddConnection,
                2 => RemoveNeuron,
                3 => MutateWeight,
                _ => MutateExponent,
            };

            // Generate a random number between 1.0 and 10.0
            let value = rng.random_range(0.0..=5.0);

            let add_to = if rng.random_bool(0.5) { -value } else { value };

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
                MutationAction::MutateExponent => {
                    self.adjust_mutate_exponent(add_to);
                }
            }

            loop_count += 1;
        }

        self.adjust_self_mutation(rng);
    }

    /// Get the overall mutation probability (0-100).
    ///
    /// This determines whether any mutations will occur at all during evolution.
    pub fn self_mutation(&self) -> u8 {
        self.self_mutation
    }

    fn adjust_self_mutation(&mut self, rng: &mut impl Rng) {
        let rate: i8 = rng.random_range(-1..=1);

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

    /// Get the normalized probability of splitting a connection (0-100).
    pub fn split_connection(&self) -> f32 {
        self.split_connection
    }

    /// Get the normalized probability of adding a new connection (0-100).
    pub fn add_connection(&self) -> f32 {
        self.add_connection
    }

    /// Get the normalized probability of removing a connection (0-100).
    pub fn remove_connection(&self) -> f32 {
        self.remove_connection
    }

    /// Get the normalized probability of mutating a weight (0-100).
    pub fn mutate_weight(&self) -> f32 {
        self.mutate_weight
    }

    /// Get the normalized probability of mutating an exponent (0-100).
    pub fn mutate_exponent(&self) -> f32 {
        self.mutate_exponent
    }

    fn adjust(&mut self, cmd: impl FnOnce(&mut Self)) {
        cmd(self);
        if self.split_connection < 0. {
            self.split_connection = 0.;
        }
        if self.add_connection < 0. {
            self.add_connection = 0.;
        }
        if self.remove_connection < 0. {
            self.remove_connection = 0.;
        }
        if self.mutate_weight < 0. {
            self.mutate_weight = 0.;
        }
        if self.mutate_exponent < 0. {
            self.mutate_exponent = 0.;
        }

        self.recalculate();
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

    fn adjust_mutate_exponent(&mut self, amt: f32) {
        self.mutate_exponent += amt;

        if self.mutate_exponent < 0. {
            self.mutate_exponent = 0.;
        }

        self.recalculate();
    }

    fn recalculate(&mut self) {
        let total = self.split_connection
            + self.add_connection
            + self.remove_connection
            + self.mutate_weight
            + self.mutate_exponent;

        self.split_connection = (self.split_connection * 100.) / total;
        self.add_connection = (self.add_connection * 100.) / total;
        self.remove_connection = (self.remove_connection * 100.) / total;
        self.mutate_weight = (self.mutate_weight * 100.) / total;
        self.mutate_exponent = (self.mutate_exponent * 100.) / total;
    }

    /// Generate a sequence of mutation actions based on the configured probabilities.
    ///
    /// This method generates multiple mutations in one go, with each subsequent
    /// mutation having reduced probability. The maximum number of mutations is
    /// limited by [`MAX_MUTATIONS`].
    ///
    /// # Arguments
    /// * `rng` - Random number generator for probabilistic selection
    ///
    /// # Returns
    /// A vector of mutation actions to apply to the network
    pub fn gen_mutation_actions(&self, rng: &mut impl Rng) -> Vec<MutationAction> {
        let mut actions = Vec::with_capacity(MAX_MUTATIONS as usize);
        let mut replica = *self;

        let mut loop_count = 0;
        while rng.gen_rate() < replica.self_mutation() && loop_count < MAX_MUTATIONS {
            let action = rng.gen_mutation_action(&replica);
            match action {
                MutationAction::SplitConnection => replica.adjust(|s| s.split_connection /= 2.),
                MutationAction::AddConnection => replica.adjust(|s| s.add_connection /= 2.),
                MutationAction::RemoveNeuron => replica.adjust(|s| s.remove_connection /= 2.),
                MutationAction::MutateWeight => replica.adjust(|s| s.mutate_weight /= 2.),
                MutationAction::MutateExponent => replica.adjust(|s| s.mutate_exponent /= 2.),
            }

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

    chances.adjust_add_connection(-10.);

    chances.adjust_remove_connection(10.);

    chances.adjust_mutate_weight(-10.);

    let total = chances.split_connection
        + chances.add_connection
        + chances.remove_connection
        + chances.mutate_weight
        + chances.mutate_exponent;
    let diff = (100. - total).abs();

    assert!(diff <= 0.0001);
}

#[test]
pub fn check_mutate() {
    let mut rng = rand::rng();

    let mut chances = MutationChances::new(50);

    for _ in 0..100 {
        chances.adjust_mutation_chances(&mut rng);

        println!("{:?}", chances);

        let total = chances.split_connection
            + chances.add_connection
            + chances.remove_connection
            + chances.mutate_weight
            + chances.mutate_exponent;

        let diff = (100. - total).abs();

        assert!(diff <= 0.0001);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_mutation_rate_ext_gen_rate() {
        let mut rng = StdRng::seed_from_u64(42);

        // Test that gen_rate produces values in range [0, 100]
        for _ in 0..1000 {
            let rate = rng.gen_rate();
            assert!(rate <= 100, "Rate {} should be <= 100", rate);
        }
    }

    #[test]
    fn test_mutation_rate_ext_gen_mutation_action() {
        let mut rng = StdRng::seed_from_u64(12345);
        let chances = MutationChances::new(100);

        // Generate many actions and ensure all types are represented
        let mut action_counts = std::collections::HashMap::new();
        for _ in 0..1000 {
            let action = rng.gen_mutation_action(&chances);
            *action_counts.entry(format!("{:?}", action)).or_insert(0) += 1;
        }

        // With equal chances, all actions should appear
        assert!(action_counts.contains_key("SplitConnection"));
        assert!(action_counts.contains_key("AddConnection"));
        assert!(action_counts.contains_key("RemoveNeuron"));
        assert!(action_counts.contains_key("MutateWeight"));
        assert!(action_counts.contains_key("MutateExponent"));
    }

    #[test]
    fn test_mutation_chances_new() {
        let chances = MutationChances::new(75);

        assert_eq!(chances.self_mutation(), 75);

        // All mutation types should have equal probability (20% each)
        assert!((chances.split_connection() - 20.0).abs() < 0.001);
        assert!((chances.add_connection() - 20.0).abs() < 0.001);
        assert!((chances.remove_connection() - 20.0).abs() < 0.001);
        assert!((chances.mutate_weight() - 20.0).abs() < 0.001);
        assert!((chances.mutate_exponent() - 20.0).abs() < 0.001);

        // Total should be 100%
        let total = chances.split_connection()
            + chances.add_connection()
            + chances.remove_connection()
            + chances.mutate_weight()
            + chances.mutate_exponent();
        assert!((total - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_mutation_chances_none() {
        let chances = MutationChances::none();

        assert_eq!(chances.self_mutation(), 0);
        assert_eq!(chances.split_connection(), 0.0);
        assert_eq!(chances.add_connection(), 0.0);
        assert_eq!(chances.remove_connection(), 0.0);
        assert_eq!(chances.mutate_weight(), 0.0);
        assert_eq!(chances.mutate_exponent(), 0.0);
    }

    #[test]
    fn test_mutation_chances_new_from_raw() {
        let chances = MutationChances::new_from_raw(80, 40.0, 30.0, 10.0, 15.0, 5.0);

        assert_eq!(chances.self_mutation(), 80);

        // Check that values are normalized to sum to 100
        let total = chances.split_connection()
            + chances.add_connection()
            + chances.remove_connection()
            + chances.mutate_weight()
            + chances.mutate_exponent();
        assert!((total - 100.0).abs() < 0.001);

        // Check relative proportions
        assert!(chances.split_connection() > chances.add_connection());
        assert!(chances.add_connection() > chances.mutate_weight());
        assert!(chances.mutate_weight() > chances.remove_connection());
        assert!(chances.remove_connection() > chances.mutate_exponent());
    }

    #[test]
    fn test_mutation_chances_new_from_raw_zero_total() {
        // Test edge case where all values are zero
        let chances = MutationChances::new_from_raw(50, 0.0, 0.0, 0.0, 0.0, 0.0);

        // Should handle division by zero gracefully
        assert_eq!(chances.self_mutation(), 50);
    }

    #[test]
    fn test_adjust_self_mutation() {
        let mut rng = StdRng::seed_from_u64(42);

        // Test boundary conditions
        let mut chances = MutationChances::new(0);
        chances.adjust_self_mutation(&mut rng);
        assert!(chances.self_mutation() <= 1); // Can only go up from 0

        let mut chances = MutationChances::new(100);
        chances.adjust_self_mutation(&mut rng);
        assert!(chances.self_mutation() >= 99); // Can only go down from 100

        // Test normal adjustment
        let mut chances = MutationChances::new(50);
        let original = chances.self_mutation();
        chances.adjust_self_mutation(&mut rng);
        let diff = (chances.self_mutation() as i32 - original as i32).abs();
        assert!(diff <= 1); // Should change by at most 1
    }

    #[test]
    fn test_individual_adjustments() {
        let mut chances = MutationChances::new(50);

        // Test positive adjustments
        chances.adjust_split_connection(10.0);
        let total = chances.split_connection()
            + chances.add_connection()
            + chances.remove_connection()
            + chances.mutate_weight()
            + chances.mutate_exponent();
        assert!((total - 100.0).abs() < 0.001);

        // Test negative adjustments
        chances.adjust_add_connection(-5.0);
        let total = chances.split_connection()
            + chances.add_connection()
            + chances.remove_connection()
            + chances.mutate_weight()
            + chances.mutate_exponent();
        assert!((total - 100.0).abs() < 0.001);

        // Test adjustment below zero
        chances.adjust_remove_connection(-100.0);
        assert!(chances.remove_connection() >= 0.0);
        let total = chances.split_connection()
            + chances.add_connection()
            + chances.remove_connection()
            + chances.mutate_weight()
            + chances.mutate_exponent();
        assert!((total - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_gen_mutation_actions() {
        let mut rng = StdRng::seed_from_u64(9876);

        // Test with high mutation chance
        let chances = MutationChances::new(90);
        let actions = chances.gen_mutation_actions(&mut rng);
        assert!(
            !actions.is_empty(),
            "Should generate some mutations with 90% chance"
        );
        assert!(actions.len() <= MAX_MUTATIONS as usize);

        // Test with zero mutation chance
        let chances = MutationChances::none();
        let actions = chances.gen_mutation_actions(&mut rng);
        assert!(
            actions.is_empty(),
            "Should generate no mutations with 0% chance"
        );

        // Test with moderate mutation chance
        let chances = MutationChances::new(50);
        let mut total_actions = 0;
        for _ in 0..100 {
            let actions = chances.gen_mutation_actions(&mut rng);
            total_actions += actions.len();
        }
        assert!(
            total_actions > 0,
            "Should generate some mutations over 100 trials"
        );
    }

    #[test]
    fn test_gen_mutation_actions_distribution() {
        let mut rng = StdRng::seed_from_u64(5555);

        // Create chances with specific distribution
        let chances = MutationChances::new_from_raw(
            100,  // Always mutate
            50.0, // High split chance
            30.0, // Medium add chance
            5.0,  // Low remove chance
            10.0, // Low weight chance
            5.0,  // Low exponent chance
        );

        // Collect many mutations
        let mut action_counts = std::collections::HashMap::new();
        for _ in 0..1000 {
            let actions = chances.gen_mutation_actions(&mut rng);
            for action in actions {
                *action_counts.entry(format!("{:?}", action)).or_insert(0) += 1;
            }
        }

        // Verify that split connections are most common
        let split_count = action_counts.get("SplitConnection").unwrap_or(&0);
        let add_count = action_counts.get("AddConnection").unwrap_or(&0);
        let remove_count = action_counts.get("RemoveNeuron").unwrap_or(&0);

        assert!(
            split_count > add_count,
            "Split should be more common than add"
        );
        assert!(
            add_count > remove_count,
            "Add should be more common than remove"
        );
    }

    #[test]
    fn test_deterministic_mutations() {
        let seed = 1111;
        let mut rng1 = StdRng::seed_from_u64(seed);
        let mut rng2 = StdRng::seed_from_u64(seed);

        let chances = MutationChances::new(75);

        // Generate actions with same seed
        let actions1 = chances.gen_mutation_actions(&mut rng1);
        let actions2 = chances.gen_mutation_actions(&mut rng2);

        // Should produce identical results
        assert_eq!(actions1.len(), actions2.len());
        for (a1, a2) in actions1.iter().zip(actions2.iter()) {
            assert_eq!(format!("{:?}", a1), format!("{:?}", a2));
        }
    }

    #[test]
    fn test_mutation_action_selection() {
        let mut rng = StdRng::seed_from_u64(7777);

        // Test extreme cases - only split connection
        let chances = MutationChances::new_from_raw(100, 100.0, 0.0, 0.0, 0.0, 0.0);
        for _ in 0..10 {
            let action = rng.gen_mutation_action(&chances);
            assert!(matches!(action, MutationAction::SplitConnection));
        }

        // Test extreme cases - only mutate exponent
        let chances = MutationChances::new_from_raw(100, 0.0, 0.0, 0.0, 0.0, 100.0);
        for _ in 0..10 {
            let action = rng.gen_mutation_action(&chances);
            assert!(matches!(action, MutationAction::MutateExponent));
        }
    }

    #[test]
    fn test_adjust_mutation_chances_evolution() {
        let mut rng = StdRng::seed_from_u64(3333);
        let mut chances = MutationChances::new(100); // High self-mutation

        let original = chances;
        chances.adjust_mutation_chances(&mut rng);

        // Should have changed something
        let changed = chances.split_connection() != original.split_connection()
            || chances.add_connection() != original.add_connection()
            || chances.remove_connection() != original.remove_connection()
            || chances.mutate_weight() != original.mutate_weight()
            || chances.mutate_exponent() != original.mutate_exponent()
            || chances.self_mutation() != original.self_mutation();

        assert!(changed, "High self-mutation should cause changes");

        // Should still sum to 100
        let total = chances.split_connection()
            + chances.add_connection()
            + chances.remove_connection()
            + chances.mutate_weight()
            + chances.mutate_exponent();
        assert!((total - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_copy_and_equality() {
        let chances1 = MutationChances::new_from_raw(75, 30.0, 25.0, 20.0, 15.0, 10.0);
        let chances2 = chances1; // Copy

        assert_eq!(chances1, chances2);
        assert_eq!(chances1.self_mutation(), chances2.self_mutation());
        assert_eq!(chances1.split_connection(), chances2.split_connection());
        assert_eq!(chances1.add_connection(), chances2.add_connection());
        assert_eq!(chances1.remove_connection(), chances2.remove_connection());
        assert_eq!(chances1.mutate_weight(), chances2.mutate_weight());
        assert_eq!(chances1.mutate_exponent(), chances2.mutate_exponent());
    }
}
