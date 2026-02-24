pub mod differential_move;
pub mod gaussian_move;

use rand::{Rng, RngCore};

use crate::{
    log_likelihood::LogLikelihoodModel,
    moves::{differential_move::DifferentialMove, gaussian_move::GaussianMove},
    state::WalkerState,
};

pub trait EnsembleMove {
    fn get_likelihood_floor(&self, rng: &mut dyn RngCore) -> f64 {
        rng.random_range(0.0_f64..1.0_f64).ln()
    }
    fn jump(
        &self,
        rng: &mut dyn RngCore,
        log_likelihood_model: &dyn LogLikelihoodModel,
        state_i: &WalkerState,
        state_j: &mut WalkerState,
    );
}

pub struct MoveHandler {
    vec_move: Vec<Box<dyn EnsembleMove>>,
    cumulative: Vec<f64>,
}

impl MoveHandler {
    pub fn new(vec_move: Vec<Box<dyn EnsembleMove>>, probability: Vec<f64>) -> Self {
        assert!(probability.len() == vec_move.len());

        let total: f64 = probability.iter().sum();

        let cumulative = probability
            .iter()
            .scan(0.0, |sum, &p| {
                *sum += p / total;
                Some(*sum)
            })
            .collect::<Vec<_>>();

        Self {
            vec_move,
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

        let i = self.cumulative.iter().position(|&c| r < c).unwrap();

        self.vec_move[i].jump(rng, log_likelihood_model, state_i, state_j);
    }
}

impl Default for MoveHandler {
    fn default() -> Self {
        Self::new(
            vec![
                Box::new(DifferentialMove::new()),
                Box::new(GaussianMove::new()),
            ],
            vec![0.9, 1.0],
        )
    }
}
