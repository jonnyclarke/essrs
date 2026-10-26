//! Module providing the different move (jump) options and handlers for managing combinations of move options

pub mod diffconv_move;
pub mod differential_move;
pub mod gaussian_move;

pub mod hybrid;

use anyhow;
use rand::{Rng, RngCore};

use crate::{log_likelihood::LogLikelihoodModel, state::WalkerState};

pub enum MoveType {
    DiffConv, // Modification of 'Differential Move' designed to enhance convergence
    Differential,
    Gaussian,

    Hybrid(HybridMoveType),
}

pub enum HybridMoveType {
    DiffNorm,
}

pub trait EnsembleMove {
    fn get_likelihood_floor<R: RngCore>(&self, rng: &mut R) -> f64 {
        rng.random_range(0.0_f64..1.0_f64).ln()
    }

    fn jump<L: LogLikelihoodModel>(
        &self,
        iteration: usize,
        log_likelihood_model: &L,
        state: &mut WalkerState,
    ) -> anyhow::Result<()>;
}
