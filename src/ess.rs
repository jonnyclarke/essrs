//! Ensemble Slice Sampler
//!
//! This is the primary API to run the MCMC process.

mod initialise;
mod new;

use crate::{
    chains::ChainBuffer,
    log_likelihood::{LogLikelihoodModel, WrappedLogLikelihoodModel},
    moves::EnsembleMoveHandler,
    state::WalkerState,
};

/// Ensemble Slice Sampler Object.
///
/// This struct manages all the internals of the MCMC fit.
/// Type is generic over:
/// * implementor of 'ChainBuffer' trait: used to store MCMC history
/// * implementor of 'MoveHandler' trait: controls how the MCMC makes proposals
/// * implementor of 'LogLikelihoodModel' trait: provides log-likelihood computation
pub struct EnsembleSliceSampler<C: ChainBuffer, M: EnsembleMoveHandler, L: LogLikelihoodModel> {
    /// the maximum number of steps for which to run the MCMC
    max_n_steps: usize,
    /// the number of walkers in the MCMC
    /// tradeoff is that more walkers provide better exploration but increase computational cost
    n_walkers: usize,
    /// number of parameters being fitted
    n_parameters: usize,

    /// chain storage object
    pub chains: C,
    /// move handler object
    pub move_handler: M,
    /// log-likelihood object
    /// this owns the data being fitted making 'EnsembleSliceSampler' generic over data
    pub model: L,

    /// walker state of the i^th iteration
    state_i: WalkerState,
    /// walker state of the j = i+1 th iteration
    /// TODO: potentially swap this for a swap storage system
    state_j: WalkerState,
}

impl<C: ChainBuffer, M: EnsembleMoveHandler, L: LogLikelihoodModel> EnsembleSliceSampler<C, M, L> {
    /// accept the state change by copying state_j into state_i
    /// TODO: remove this for swap style storage
    pub fn accept_proposed_state(&mut self) {
        // let mut state0 = self.state_i.get_mut_state_matrix();
        self.state_i
            .get_mut_state_matrix()
            .assign(&self.state_j.get_state_matrix());

        // let mut ll0 = self.state_i.get_mut_ll_vector();
        self.state_i
            .get_mut_ll_vector()
            .assign(&self.state_j.get_mut_ll_vector());
    }

    /// Run the MCMC sampler
    ///
    /// # Arguments
    /// * n_burn_in: the number of iterations to allow walker convergence prior to storing samples
    ///
    /// # TODO
    /// * Upgrade ChainBuffer trait to provide a method to update itself
    pub fn run_sampler(&mut self, n_burn_in: usize) -> anyhow::Result<()> {
        // we allow burn in to eliminate effect on chains of the starting point
        for iteration in 1..=n_burn_in {
            self.move_handler.distribute_jump(
                iteration,
                &self.model,
                &self.state_i,
                &mut self.state_j,
            )?;
            self.accept_proposed_state(); // we accept new state before storing to avoid storing the initial conditions state...
        }

        for iteration in 1..=self.max_n_steps {
            self.move_handler.distribute_jump(
                iteration,
                &self.model,
                &self.state_i,
                &mut self.state_j,
            )?;
            self.accept_proposed_state(); // we accept new state before storing to avoid storing the initial conditions state...

            let internal = self.state_i.get_state_matrix();
            let mut physical = internal.to_owned();

            self.model
                .wrapped_columnar_transform_internal_to_physical(internal, physical.view_mut())?;
            self.chains.record_state(&physical);
        }

        Ok(())
    }

    /// Wrapper function to call the chain based function displaying parameter summaries
    pub fn display_parameter_summaries(&self) {
        self.chains
            .display_parameter_summaries(self.n_parameters as i32);
    }

    // TODO: remove this function; pass as standalone functions
    pub fn print_quantile_summary(&self) -> anyhow::Result<()> {
        // Make a sorted copy
        let mut values = self.state_i.get_ll_vector().to_vec();
        values.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("sorting of log-likelihood vector has failed")
        });

        let n = values.len();

        let p25 = n / 4;
        let p50 = n / 2;
        let p75 = 3 * n / 4;

        for (i, v) in values.iter().enumerate() {
            if i == 0 {
                println!("-- MIN: {}", v)
            }
            if i == p25 {
                println!("-- p25: {}", v)
            }
            if i == p50 {
                println!("-- p50: {}", v)
            }
            if i == p75 {
                println!("-- p75: {}", v)
            }
            if i == n - 1 {
                println!("-- MAX: {}\n", v)
            }
        }
        Ok(())
    }
}
