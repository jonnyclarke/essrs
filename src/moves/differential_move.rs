//! This is the core jump algorithm used in conjunction with ensemble slice sampling.
//!
//! The core algorithm is as follows:
//! | Walker `i` starts with parameter vector v0 and a likelihood l0.
//! | We sample two additional and different parameter vectors, v1 and v2.
//! | The difference between these two vectors then defines a direction vector: d = v2 - v1
//! | We sample a random number in range 0 to 1 and take logarithm; this defines the acceptance boundary `l_b` = l0 + ln(rn)
//! | We step out; we progressively test points in both directions, d and -d, from v0 until the log-likelihood falls below the acceptance boundary
//! | We step in; we randomly sample within the range [v0 - n * d, v0 + m * d] until, at (v0', l0') the log-likelihood is greater than the acceptance threshold.
//! | The new point v0', and corresponding likelihood, l0', are stored as next element on the MCMC walk.

use ndarray::{Axis, parallel::prelude::*};

use crate::{
    log_likelihood::LogLikelihoodModel,
    moves::{DynEnsembleMove, EnsembleMove, compute_likelihood_floor},
    random::{get_rn_not, get_rn_not_or},
    slice::{step_in, step_out},
    state::WalkerState,
    threading::Threading,
};

/// Differential Move
#[derive(Debug, PartialEq)]
pub struct DifferentialMove {}

impl DifferentialMove {
    /// Constructor
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
    fn jump_implementation<L>(
        &self,
        _iteration: usize,
        log_likelihood_model: &L,
        state: &mut WalkerState,
        threads: &Threading,
    ) -> anyhow::Result<()>
    where
        L: LogLikelihoodModel + Sync + ?Sized,
    {
        let n_walkers = state.n_walkers();

        let (current, ll_current, mut next, mut ll_next) = state.parallel_buffers();

        threads.pool.install(|| {
            let _ = current
                .axis_iter(Axis(0))
                .into_par_iter()
                .zip(ll_current.axis_iter(Axis(0)).into_par_iter())
                .zip(next.axis_iter_mut(Axis(0)).into_par_iter())
                .zip(ll_next.axis_iter_mut(Axis(0)).into_par_iter())
                .enumerate()
                .try_for_each(
                    |(i, (((current_row, current_ll), mut next_row), mut next_ll))| {
                        let worker_index = threads
                            .pool
                            .current_thread_index()
                            .expect("NOT running inside the sampler thread pool");

                        let _ = threads.with_worker(worker_index, |worker| {
                            let rng = &mut worker.rng;
                            let direction = &mut worker.scratch.direction;
                            let internal = &mut worker.scratch.internal;
                            let physical = &mut worker.scratch.physical;

                            let ll_floor = compute_likelihood_floor(current_ll[()], rng);

                            let l = get_rn_not(rng, n_walkers, i);
                            let r = get_rn_not_or(rng, n_walkers, i, l);

                            ndarray::Zip::from(direction.view_mut())
                                .and(&current.row(l))
                                .and(&current.row(r))
                                .for_each(|out, &left, &right| {
                                    *out = left - right;
                                });

                            let search_bounds = step_out(
                                log_likelihood_model,
                                current_row.view(),
                                direction.view(),
                                &mut internal.view_mut(),
                                &mut physical.view_mut(),
                                ll_floor,
                            )?;

                            let log_likelihood = step_in(
                                rng,
                                log_likelihood_model,
                                current_row.view(),
                                direction.view(),
                                search_bounds,
                                &mut internal.view_mut(),
                                &mut physical.view_mut(),
                            )?;

                            // Write directly to this walker's output buffers.
                            next_row.assign(&internal.view());
                            next_ll[()] = log_likelihood;

                            Ok::<(), anyhow::Error>(())
                        });
                        Ok::<(), anyhow::Error>(())
                    },
                );
        });

        Ok(())
    }
}

impl EnsembleMove for DifferentialMove {
    fn jump<L>(
        &self,
        iteration: usize,
        log_likelihood_model: &L,
        state: &mut WalkerState,
        threads: &Threading,
    ) -> anyhow::Result<()>
    where
        L: LogLikelihoodModel + Sync + ?Sized,
    {
        self.jump_implementation(iteration, log_likelihood_model, state, threads)
    }
}

impl DynEnsembleMove for DifferentialMove {
    fn jump_dyn(
        &self,
        iteration: usize,
        log_likelihood_model: &(dyn LogLikelihoodModel + Sync),
        state: &mut WalkerState,
        threads: &Threading,
    ) -> anyhow::Result<()> {
        self.jump_implementation(iteration, log_likelihood_model, state, threads)
    }
}

#[cfg(test)]
mod tests {

    use ndarray::array;

    use super::*;
    use crate::testing::{TestLeftRightUniformLogL, WalkerStateBuilder};

    #[test]
    fn test_initialisation() {
        let _ = DifferentialMove::new();
        let _ = DifferentialMove::default();
    }

    #[test]
    fn test_jump_implementation() -> anyhow::Result<()> {
        let model = TestLeftRightUniformLogL::default();

        let mut state = WalkerStateBuilder::new(3, 1)
            .set_state_matrix(array![[-2.0], [1.3], [3.45]])
            .set_ll_vector(array![model.get_p(), model.get_p(), model.get_p()])
            .swap()
            .construct();

        println!("{:#?}", state);

        let threads = Threading::new(Some(1), 1)?;

        let diff_move = DifferentialMove::new();
        diff_move.jump_implementation(1, &model, &mut state, &threads)?;

        state.swap();

        for i in 0..3 {
            assert!(state.get_ith_ll_value(i).eq(&model.get_p()));

            let x = state.get_ith_state_vector_view(i)[0];
            let (l, r) = model.get_lr();
            assert!(x.gt(&l) && x.lt(&r))
        }

        Ok(())
    }
}
