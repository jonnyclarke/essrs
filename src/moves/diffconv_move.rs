//! # Warning
//! This move should not be used for posterior sampling.
//! This move is only designed to help in the convergence stage by limiting directions to be towards other walkers.

use ndarray::{Array1, ArrayViewMut1, Zip};
use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;

use crate::{
    log_likelihood::LogLikelihoodModel,
    moves::{DynEnsembleMove, EnsembleMove, compute_likelihood_floor},
    random::get_rn_not,
    slice::{step_in, step_out},
    state::WalkerState,
};

#[derive(Debug, PartialEq)]
pub struct DiffConvMove {}

impl DiffConvMove {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for DiffConvMove {
    fn default() -> Self {
        Self::new()
    }
}

impl DiffConvMove {
    fn get_vector(
        &self,
        state: &WalkerState,
        l: usize,
        r: usize,
        direction: &mut ArrayViewMut1<f64>,
    ) {
        Zip::from(&mut *direction)
            .and(&state.get_ith_state_vector_view(l))
            .and(&state.get_ith_state_vector_view(r))
            .for_each(|out, &left, &right| {
                *out = left - right;
            });
    }
}

impl DiffConvMove {
    fn jump_implementation<L>(
        &self,
        _iteration: usize,
        log_likelihood_model: &L,
        state: &mut WalkerState,
    ) -> anyhow::Result<()>
    where
        L: LogLikelihoodModel + Sync + ?Sized,
    {
        let n_walkers = state.n_walkers();

        let proposals: Vec<anyhow::Result<(Array1<f64>, f64)>> = (0..n_walkers)
            .into_par_iter()
            .map(|i| {
                let mut rng: StdRng = StdRng::from_rng(&mut rand::rng());

                // Each worker owns its own scratch arrays.
                let mut direction = state.get_ith_state_vector_view(i).to_owned();

                let mut internal = state.get_ith_state_vector_view(i).to_owned();
                let mut physical = state.get_ith_state_vector_view(i).to_owned();

                let ll_floor = compute_likelihood_floor(state.get_ith_ll_value(i), &mut rng);

                let current = state.get_ith_state_vector_view(i);

                let t = get_rn_not(&mut rng, n_walkers, i);

                self.get_vector(state, i, t, &mut direction.view_mut());

                let search_bounds = step_out(
                    log_likelihood_model,
                    current.view(),
                    direction.view(),
                    &mut internal.view_mut(),
                    &mut physical.view_mut(),
                    ll_floor,
                )?;

                let log_likelihood = step_in(
                    &mut rng,
                    log_likelihood_model,
                    current.view(),
                    direction.view(),
                    search_bounds,
                    &mut internal.view_mut(),
                    &mut physical.view_mut(),
                )?;

                Ok((internal, log_likelihood))
            })
            .collect();

        // Commit the results sequentially.
        for (i, proposal) in proposals.into_iter().enumerate() {
            let (internal, log_likelihood) = proposal?;

            state.assign_ith_state_vector(i, internal.view());
            state.assign_ith_ll(i, log_likelihood);
        }

        Ok(())
    }
}

impl EnsembleMove for DiffConvMove {
    fn jump<L>(
        &self,
        iteration: usize,
        log_likelihood_model: &L,
        state: &mut WalkerState,
    ) -> anyhow::Result<()>
    where
        L: LogLikelihoodModel + Sync + ?Sized,
    {
        self.jump_implementation(iteration, log_likelihood_model, state)
    }
}

impl DynEnsembleMove for DiffConvMove {
    fn jump_dyn(
        &self,
        iteration: usize,
        log_likelihood_model: &(dyn LogLikelihoodModel + Sync),
        state: &mut WalkerState,
    ) -> anyhow::Result<()> {
        self.jump_implementation(iteration, log_likelihood_model, state)
    }
}

// #[cfg(test)]
// mod tests {

//     // use anyhow;
//     // use approx::assert_relative_eq;
//     // use rstest::rstest;
//     use ndarray::array;

//     use super::*;
//     use crate::testing::WalkerStateBuilder;

//     #[test]
//     fn test_get_vector() {
//         let state = WalkerStateBuilder::new(3, 1)
//             .set_state_matrix(array![[-1.0], [0.0], [1.0]])
//             .set_ll_vector(array![-1.0, 0.0, 1.0])
//             .swap()
//             .construct();

//         let diff_move = DifferentialMove::default();

//         let mut direction = array![0.0];
//         diff_move.get_vector(&state, 2, 1, &mut direction.view_mut());

//         assert_eq!(direction, array![1.0])
//     }
// }
