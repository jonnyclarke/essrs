//! Ensemble Slice Sampler
//!
//! This is the primary API to run the MCMC process.

mod initialise;
mod new;

use ndarray::Array3;

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
    chains: C,
    /// move handler object
    move_handler: M,
    /// log-likelihood object
    /// this owns the data being fitted making 'EnsembleSliceSampler' generic over data
    model: L,

    /// walker state: includes 'current' and 'next'
    state: WalkerState,
}

impl<C: ChainBuffer, M: EnsembleMoveHandler, L: LogLikelihoodModel> EnsembleSliceSampler<C, M, L> {
    /// Calls state-buffer swap which flips current and next memory locations
    pub fn accept_proposed_state(&mut self) {
        self.state.swap();
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
                &mut self.state,
            )?;
            self.accept_proposed_state(); // we accept new state before storing to avoid storing the initial conditions state...
        }

        for iteration in 1..=self.max_n_steps {
            self.move_handler.distribute_jump(
                iteration,
                &self.model,
                &mut self.state,
            )?;
            self.accept_proposed_state(); // we accept new state before storing to avoid storing the initial conditions state...

            let internal = self.state.get_state_matrix_view();
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

}
