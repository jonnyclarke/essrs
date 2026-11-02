//! Ensemble slice move implementations.

pub mod diffconv_move;
pub mod differential_move;

pub mod hybrid;

use anyhow;
use rand::{Rng, rngs::StdRng};

use crate::{log_likelihood::LogLikelihoodModel, state::WalkerState, threading::Threading};

/// Compute the likelihood floor above which proposals can be accepted.
///
/// This enables exploration of full parameter space by setting the acceptance threshold slightly lower than the current log-likelihood.
/// Values are sampled in range [0, 1] thus increment ranges from [-INF, 0]
///
/// # Arguments:
/// * ll: current log-likelihood
/// * rng: random number generator from which is sampled the increment
pub fn compute_likelihood_floor(ll: f64, rng: &mut StdRng) -> f64 {
    ll + rng.random_range(0.0_f64..=1.0_f64).ln()
}
/// Trait definition for a static-dispatch Ensemble Move implementation
pub trait EnsembleMove {
    /// Jump function; defines how a moves explores parameter space
    ///
    /// # Errors
    /// * failure of 'jump' implementation
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

/// Trait definition for a dynamic-dispatch Ensemble Move implementation
pub trait DynEnsembleMove: Send + Sync {
    /// Jump function; defines how a moves explores parameter space
    ///
    /// # Arguments
    /// * iteration: iteration of the MCMC process (used in hybrid moves such as a 'one-in-k' style scheme).
    ///
    /// # Errors
    /// * failure of `jump_dyn` implementation
    fn jump_dyn(
        &self,
        iteration: usize,
        log_likelihood_model: &(dyn LogLikelihoodModel + Sync),
        state: &mut WalkerState,
        threads: &Threading,
    ) -> anyhow::Result<()>;
}
