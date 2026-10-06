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

use ndarray::{ArrayViewMut1, Zip};
use rand::Rng;

use crate::{
    log_likelihood::LogLikelihoodModel,
    moves::EnsembleMove,
    random::{get_rn_not, get_rn_not_or},
    slice::{step_in, step_out},
    state::WalkerState,
};
#[derive(Debug, PartialEq)]
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
    fn get_vector(
        &self,
        state: &WalkerState,
        l: usize,
        r: usize,
        direction: &mut ArrayViewMut1<f64>,
    ) {
        Zip::from(&mut *direction)
            .and(&state.get_ith_state_vector(l))
            .and(&state.get_ith_state_vector(r))
            .for_each(|out, &left, &right| {
                *out = left - right;
            });
    }
}

impl EnsembleMove for DifferentialMove {
    fn jump<L: LogLikelihoodModel, R: Rng>(
        &self,
        rng: &mut R,
        log_likelihood_model: &L,
        state_i: &WalkerState,
        state_j: &mut WalkerState,
    ) -> anyhow::Result<()> {
        let n_walkers = state_i.n_walkers();

        // we generate multiple scratch arrays to avoid constantly re-allocating
        let mut direction = state_i.get_ith_state_vector(0).to_owned();
        let mut internal = state_i.get_ith_state_vector(0).to_owned();
        let mut physical = state_i.get_ith_state_vector(0).to_owned();

        for i in 0..n_walkers {
            let ll_floor = state_i.get_ith_ll(i) + self.get_likelihood_floor(rng);

            let current = state_i.get_ith_state_vector(i);

            let l = get_rn_not(rng, n_walkers, i);
            let r = get_rn_not_or(rng, n_walkers, i, l);
            self.get_vector(state_i, l, r, &mut direction.view_mut());

            let search_bounds = step_out(
                log_likelihood_model,
                current.view(),
                direction.view(),
                &mut internal.view_mut(),
                &mut physical.view_mut(),
                ll_floor,
            )?;

            let log_likelihood = step_in(
                rng,
                log_likelihood_model,
                current.view(),
                direction.view(),
                search_bounds,
                &mut internal.view_mut(),
                &mut physical.view_mut(),
            )?;

            state_j.get_mut_ith_state_vector(i).assign(&internal);
            *state_j.get_mut_ith_ll(i) = log_likelihood;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {

    // use anyhow;
    // use approx::assert_relative_eq;
    // use rstest::rstest;
    use ndarray::array;

    use super::*;
    use crate::testing::WalkerStateBuilder;

    #[test]
    fn test_get_vector() {
        let state = WalkerStateBuilder::new()
            .set_state_matrix(&array![[-1.0], [0.0], [1.0]])
            .set_ll_array(&array![-1.0, 0.0, 1.0])
            .construct();

        let diff_move = DifferentialMove::default();

        let mut direction = array![0.0];
        diff_move.get_vector(&state, 2, 1, &mut direction.view_mut());

        assert_eq!(direction, array![1.0])
    }
}
