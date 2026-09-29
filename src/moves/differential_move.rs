//! This is the core jump algorithm used in conjunction with ensemble slice sampling.
//!
//! The core algorithm is as follows:
//! | Walker `i` starts with parameter vector v0 and a likelihood l0.
//! | We sample two additional and different parameter vectors, v1 and v2.
//! | The difference between these two vectors then defines a direction vector: d = v2 - v1
//! | We sample a random number in range 0 to 1 and take logarithm; this defines the acceptance boundary l_b = l0 + ln(rn)
//! | We step out; we progressively test points in both directions, d and -d, from v0 until the log-likelihood falls below the acceptance boundary
//! | We step in; we randomly sample within the range [v0 - n * d, v0 + m * d] until, at (v0', l0') the log-likelihood is greater than the acceptance threshold.
//! | The new point v0', and corresponding likelihood, l0', are stored as next element on the MCMC walk.

use ndarray::Array1;
use rand::RngCore;

use crate::{
    log_likelihood::LogLikelihoodModel,
    moves::EnsembleMove,
    random::{get_rn_not, get_rn_not_or},
    slice::{step_in, step_out},
    state::WalkerState,
};

pub struct DifferentialMove {}
impl DifferentialMove {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for DifferentialMove {
    fn default() -> Self {
        Self::new()
    }
}

impl DifferentialMove {
    fn get_vector(&self, state: &WalkerState, l: usize, r: usize) -> Array1<f64> {
        &state.get_ith_state_vector(l) - &state.get_ith_state_vector(r)
    }
}

impl EnsembleMove for DifferentialMove {
    fn jump(
        &self,
        rng: &mut dyn RngCore,
        log_likelihood_model: &dyn LogLikelihoodModel,
        state_i: &WalkerState,
        state_j: &mut WalkerState,
    ) {
        let n_walkers = state_i.n_walkers();

        for i in 0..n_walkers {
            let ll_floor = state_i.get_ith_ll(i) + self.get_likelihood_floor(rng);

            let current = state_i.get_ith_state_vector(i).to_owned();

            let l = get_rn_not(rng, n_walkers, i);
            let r = get_rn_not_or(rng, n_walkers, i, l);
            let direction = self.get_vector(state_i, l, r);

            let search_bounds = step_out(
                log_likelihood_model,
                current.view(),
                direction.view(),
                ll_floor,
            );

            let (ll, accepted) = step_in(
                rng,
                log_likelihood_model,
                current.view(),
                direction.view(),
                search_bounds,
                ll_floor,
            );

            state_j.get_mut_ith_state_vector(i).assign(&accepted);
            *state_j.get_mut_ith_ll(i) = ll;
        }
    }
}
