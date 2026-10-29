//! # Warning
//! This move should not be used for posterior sampling.
//! This move is only designed to help in the convergence stage by limiting directions to be towards other walkers.

use ndarray::{Axis, parallel::prelude::*};

use crate::{
    log_likelihood::LogLikelihoodModel,
    moves::{DynEnsembleMove, EnsembleMove, compute_likelihood_floor},
    random::get_rn_not,
    slice::{step_in, step_out},
    state::WalkerState,
    threading::Threading,
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

                            let t = get_rn_not(rng, n_walkers, i);

                            ndarray::Zip::from(direction.view_mut())
                                .and(&current.row(i))
                                .and(&current.row(t))
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

impl EnsembleMove for DiffConvMove {
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

impl DynEnsembleMove for DiffConvMove {
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
