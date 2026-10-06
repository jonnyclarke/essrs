//! Module providing the different move (jump) options and handlers for managing combinations of move options

pub mod move_handlers;
pub mod slice_sampler;

use anyhow;
use rand::{Rng, RngCore};

use crate::{
    log_likelihood::WrappedLogLikelihoodModel,
    moves::slice_sampler::differential_move::DifferentialMove, state::WalkerState,
};

pub enum MoveType {
    Differential,
    Gaussian,
}

pub trait EnsembleMove {
    fn get_likelihood_floor(&self, rng: &mut dyn RngCore) -> f64 {
        rng.random_range(0.0_f64..1.0_f64).ln()
    }
    fn jump<L: WrappedLogLikelihoodModel>(
        &self,
        log_likelihood_model: &L,
        state_i: &WalkerState,
        state_j: &mut WalkerState,
    ) -> anyhow::Result<()>;
}

pub trait EnsembleMoveHandler {
    fn choose_move(&self, iteration: usize) -> MoveType;

    fn distribute_jump<L: WrappedLogLikelihoodModel>(
        &mut self,
        iteration: usize,
        model: &L,
        state_i: &WalkerState,
        state_j: &mut WalkerState,
    ) -> anyhow::Result<()>;
}
