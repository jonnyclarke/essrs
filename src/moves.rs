//! Ensemble slice move implementations.

pub mod diffconv_move;
pub mod differential_move;

pub mod hybrid;

use anyhow;
use rand::{Rng, rngs::StdRng};

use crate::{log_likelihood::LogLikelihoodModel, state::WalkerState, threading::Threading};

pub enum MoveType {
    DiffConv, // Modification of 'Differential Move' designed to enhance convergence
    Differential,
    Gaussian,

    Hybrid(HybridMoveType),
}

pub enum HybridMoveType {
    DiffNorm,
}

pub fn compute_likelihood_floor(ll: f64, rng: &mut StdRng) -> f64 {
    ll + rng.random_range(0.0_f64..1.0_f64).ln()
}

pub trait EnsembleMove {
    fn jump<L>(
        &self,
        iteration: usize,
        log_likelihood_model: &L,
        state: &mut WalkerState,
        threads: &Threading,
    ) -> anyhow::Result<()>
    where
        L: LogLikelihoodModel + Sync + ?Sized;
}

pub trait DynEnsembleMove: Send + Sync {
    fn jump_dyn(
        &self,
        iteration: usize,
        log_likelihood_model: &(dyn LogLikelihoodModel + Sync),
        state: &mut WalkerState,
        threads: &Threading,
    ) -> anyhow::Result<()>;
}
