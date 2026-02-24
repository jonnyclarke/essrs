pub mod differential_move;
pub mod gaussian_move;

use rand::Rng;

use crate::{
    log_likelihood::LogLikelihoodModel,
    moves::{differential_move::DifferentialMove, gaussian_move::GaussianMove},
    state::WalkerState,
};

pub trait EnsembleMove {
    fn get_likelihood_floor(&self, rng: &mut impl Rng) -> f64 {
        rng.random_range(0.0_f64..1.0_f64).ln()
    }
    fn jump(
        &self,
        rng: &mut impl Rng,
        log_likelihood_model: &dyn LogLikelihoodModel,
        state_i: &WalkerState,
        state_j: &mut WalkerState,
    );
}

pub struct MoveHandler {
    differential: DifferentialMove,
    gaussian: GaussianMove,
    cumulative: Vec<f64>,
}

impl MoveHandler {
    pub fn new(probability: Vec<f64>) -> Self {
        assert!(probability.len() == 2);

        let total: f64 = probability.iter().sum();

        let cumulative = probability
            .iter()
            .scan(0.0, |sum, &p| {
                *sum += p / total;
                Some(*sum)
            })
            .collect::<Vec<_>>();

        Self {
            differential: DifferentialMove::new(),
            gaussian: GaussianMove::new(),
            cumulative,
        }
    }

    pub fn distribute_jump(
        &self,
        rng: &mut impl Rng,
        log_likelihood_model: &dyn LogLikelihoodModel,
        state_i: &WalkerState,
        state_j: &mut WalkerState,
    ) {
        let r: f64 = rng.random();

        match self.cumulative.iter().position(|&c| r < c) {
            Some(0) => self
                .differential
                .jump(rng, log_likelihood_model, state_i, state_j),
            Some(1) => self
                .gaussian
                .jump(rng, log_likelihood_model, state_i, state_j),
            _ => panic!("Invalid probability selection!"),
        }
    }
}
