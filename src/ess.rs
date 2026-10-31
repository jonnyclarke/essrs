//! Primary API of the Ensemble Slice Sampler MCMC process.

use ndarray::{Array2, Array3, ArrayView2};
use thiserror::Error;

use crate::{
    chains::{ChainBuffer, static_buffer::StaticBuffer},
    log_likelihood::{LogLikelihoodError, LogLikelihoodModel, WrappedLogLikelihoodModel},
    moves::{DynEnsembleMove, EnsembleMove},
    state::WalkerState,
    threading::Threading,
};

/// Error enum for the EnsembleSliceSampler struct
#[derive(Debug, Error)]
pub enum InitialisationError {
    /// initialisation fails due to invalid log-likelihood
    #[error("initialisation of initial state has failed: {0}")]
    InvalidLogLikelihood(#[from] LogLikelihoodError),
}

/// Ensemble Slice Sampler Object.
///
/// This struct manages all the internals of the MCMC fit.
/// Type is generic over:
/// * implementor of 'MoveHandler' trait: controls how the MCMC makes proposals
/// * implementor of 'LogLikelihoodModel' trait: provides log-likelihood computation
pub struct EnsembleSliceSampler {
    /// the maximum number of steps for which to run the MCMC
    n_samples: usize,
    /// the number of walkers in the MCMC
    /// tradeoff is that more walkers provide better exploration but increase computational cost
    n_walkers: usize,
    /// number of parameters being fitted
    n_parameters: usize,
    /// number of steps per thinning; i.e. take the last state out of n
    n_thinning_per_sample: usize,

    /// chain storage object
    chains: StaticBuffer,

    /// walker state: includes 'current' and 'next'
    state: WalkerState,

    /// threading
    threads: Threading,
}

impl EnsembleSliceSampler {
    /// Generate a new EnsembleSliceSampler object
    ///
    /// # Arguments
    /// * n_samples: the number of steps the sampler will take
    /// * n_walkers: the number of walkers used for parallel sampling
    /// * n_parameters: the number of parameters in the likelihood model
    pub fn new(
        n_samples: usize,
        n_walkers: usize,
        n_parameters: usize,
        n_threads: Option<usize>,
        n_thinning_per_sample: Option<usize>,
    ) -> Self {
        let required_walkers = (n_parameters * 2).max(3);
        assert!(n_walkers >= required_walkers);

        let threads = Threading::new(n_threads, n_parameters).unwrap();

        Self {
            n_samples,
            n_walkers,
            n_parameters,
            n_thinning_per_sample: n_thinning_per_sample.unwrap_or(1),

            chains: StaticBuffer::new(n_samples, n_walkers, n_parameters),
            state: WalkerState::new(n_walkers, n_parameters),
            threads,
        }
    }

    /// Calls state-buffer swap which flips current and next memory locations
    pub fn accept_proposed_state(&mut self) {
        self.state.swap();
    }

    /// Wrapper function to call the chain based function displaying parameter summaries
    pub fn display_parameter_summaries(&self) {
        self.chains
            .display_parameter_summaries(self.n_parameters as i32);
    }

    /// Extract full state history
    ///
    /// This wraps the chain buffer object's 'extract_state_history' function
    pub fn extract_state_history(&self) -> Array3<f64> {
        self.chains.extract_state_history()
    }

    /// Extract full log likelihood history
    ///
    /// This wraps the chain buffer object's 'extract_state_history' function
    pub fn extract_log_likelihood_history(&self) -> Array2<f64> {
        self.chains.extract_log_likeihood_history()
    }

    /// Initialise from physical initial conditions
    ///
    /// # Arguments:
    /// * model: log-likelihood model
    /// * physical: Array2 object [n_walker, n_parameter] containing initial conditions in physical parameter space
    pub fn initialise<L>(
        &mut self,
        model: &L,
        physical: ArrayView2<f64>,
    ) -> Result<(), InitialisationError>
    where
        L: LogLikelihoodModel + ?Sized,
    {
        let mut internal = physical.to_owned();

        model.wrapped_columnar_transform_physical_to_internal(physical, internal.view_mut())?;

        // this is used simply to pass to log-likelihood in case we need to report an error
        // TODO: can get rid of this allocation?
        let mut physical = self.state.get_ith_state_vector_view(0).to_owned();

        for i in 0..self.n_walkers {
            self.state.assign_ith_ll(
                i,
                model.wrapped_log_likelihood(internal.row(i), &mut physical.view_mut())?,
            )
        }

        self.state.assign_state_matrix(internal.view());

        // we have been working with next. swap next -> current
        self.state.swap();

        Ok(())
    }

    /// Initialise from physical initial conditions -- dynamic dispatch models
    ///
    /// # Arguments:
    /// * model: log-likelihood model
    /// * physical: Array2 object [n_walker, n_parameter] containing initial conditions in physical parameter space
    pub fn initialise_dyn(
        &mut self,
        model: &dyn LogLikelihoodModel,
        physical: ArrayView2<f64>,
    ) -> Result<(), InitialisationError> {
        self.initialise(model, physical)
    }

    /// Run burn in phase to converge walkers to posterior probability region
    pub fn run_burn_in<L, M>(
        &mut self,
        model: &L,
        burn_move_handler: &M,
        n_burn_in: usize,
    ) -> anyhow::Result<()>
    where
        L: LogLikelihoodModel + ?Sized,
        M: EnsembleMove + ?Sized,
    {
        for iteration in 1..=n_burn_in {
            burn_move_handler.jump(iteration, model, &mut self.state, &self.threads)?;
            self.accept_proposed_state(); // we accept new state before storing to avoid storing the initial conditions state...
        }

        Ok(())
    }

    /// Run burn in phase -- dynamic dispatch mode
    pub fn run_burn_in_dyn(
        &mut self,
        model: &dyn LogLikelihoodModel,
        burn_move_handler: &dyn DynEnsembleMove,
        n_burn_in: usize,
    ) -> anyhow::Result<()> {
        for iteration in 1..=n_burn_in {
            burn_move_handler.jump_dyn(iteration, model, &mut self.state, &self.threads)?;
            self.accept_proposed_state(); // we accept new state before storing to avoid storing the initial conditions state...
        }

        Ok(())
    }

    /// Move state into chain storage
    ///
    /// # Arguments:
    /// * model: this is required to apply the columnar internal-to-physical transform
    fn state_to_chains<L>(&mut self, model: &L) -> anyhow::Result<()>
    where
        L: LogLikelihoodModel + ?Sized,
    {
        self.chains
            .get_next_log_likelihood_vec()
            .assign(&self.state.get_ll_vector_view());

        let internal = self.state.get_state_matrix_view();
        model.wrapped_columnar_transform_internal_to_physical(
            internal,
            self.chains.get_next_state_matrix(),
        )?;

        self.chains.increment_n_stored();
        Ok(())
    }

    /// Run the MCMC sampler -- static dispatch
    ///
    /// Uses static dispatch for optimised applications
    ///
    /// # Arguments
    /// * model: the log-likelihood model to search over
    /// * move_handler: the move handler used for progressing each walker
    pub fn run_sampler<L: LogLikelihoodModel, M: EnsembleMove>(
        &mut self,
        model: &L,
        move_handler: &M,
    ) -> anyhow::Result<()> {
        for iteration in 1..=self.n_samples * self.n_thinning_per_sample {
            move_handler.jump(iteration, model, &mut self.state, &self.threads)?;
            self.accept_proposed_state();

            if iteration.is_multiple_of(self.n_thinning_per_sample) {
                self.state_to_chains(model)?;
            }
        }

        Ok(())
    }

    /// Run the MCMC sampler -- dynamic dispatch
    ///
    /// Uses dynamic dispatch for greater flexibility
    /// e.g. python bindings
    ///
    /// # Arguments
    /// * model: the log-likelihood model to search over
    /// * move_handler: the move handler used for progressing each walker
    pub fn run_sampler_dyn(
        &mut self,
        model: &dyn LogLikelihoodModel,
        move_handler: &dyn DynEnsembleMove,
    ) -> anyhow::Result<()> {
        for iteration in 1..=self.n_samples * self.n_thinning_per_sample {
            move_handler.jump_dyn(iteration, model, &mut self.state, &self.threads)?;
            self.accept_proposed_state();

            if iteration.is_multiple_of(self.n_thinning_per_sample) {
                self.state_to_chains(model)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests_ess {
    use anyhow;
    use approx::assert_relative_eq;
    use ndarray::array;

    use super::*;
    use crate::{
        functions::transforms::softplus::softplus_inverse,
        log_likelihood::f_scientific::GaussianLl1dDataErrors,
    };

    #[test]
    fn test_initialise() -> anyhow::Result<()> {
        let data = array![[0.0_f64, 0.0_f64]];
        let model = GaussianLl1dDataErrors::new(&data);

        let mut ess = EnsembleSliceSampler::new(2, 4, 2, None, None);

        let physical = array![
            [0.0_f64, 1.0_f64],
            [1.0_f64, 1.0_f64],
            [-1.0_f64, 1.0_f64],
            [2.0_f64, 1.0_f64]
        ];
        ess.initialise(&model, physical.view())?;

        let expected_ll = array![
            (softplus_inverse(1.0_f64)? - 1.0)
                + (0.0_f64.exp() / (2.0 * std::f64::consts::PI).sqrt()).ln(),
            (softplus_inverse(1.0_f64)? - 1.0)
                + ((-0.5_f64).exp() / (2.0 * std::f64::consts::PI).sqrt()).ln(),
            (softplus_inverse(1.0_f64)? - 1.0)
                + ((-0.5_f64).exp() / (2.0 * std::f64::consts::PI).sqrt()).ln(),
            (softplus_inverse(1.0_f64)? - 1.0)
                + ((-2.0_f64).exp() / (2.0 * std::f64::consts::PI).sqrt()).ln(),
        ];

        for (computed, expected) in ess
            .state
            .get_ll_vector_view()
            .iter()
            .zip(expected_ll.iter())
        {
            assert_relative_eq!(computed, expected, epsilon = 1e-6);
        }

        let true_data_state = array![
            [0.0_f64, softplus_inverse(1.0_f64)?],
            [1.0_f64, softplus_inverse(1.0_f64)?],
            [-1.0_f64, softplus_inverse(1.0_f64)?],
            [2.0_f64, softplus_inverse(1.0_f64)?]
        ];
        assert_relative_eq!(
            true_data_state,
            ess.state.get_state_matrix_view(),
            epsilon = 1e-6
        );

        Ok(())
    }

    #[test]
    fn test_initialise_dyn() -> anyhow::Result<()> {
        let data = array![[0.0_f64, 0.0_f64]];
        let model = GaussianLl1dDataErrors::new(&data);

        let mut ess = EnsembleSliceSampler::new(2, 4, 2, None, None);

        let physical = array![
            [0.0_f64, 1.0_f64],
            [1.0_f64, 1.0_f64],
            [-1.0_f64, 1.0_f64],
            [2.0_f64, 1.0_f64]
        ];
        ess.initialise_dyn(&model, physical.view())?;

        let expected_ll = array![
            (softplus_inverse(1.0_f64)? - 1.0)
                + (0.0_f64.exp() / (2.0 * std::f64::consts::PI).sqrt()).ln(),
            (softplus_inverse(1.0_f64)? - 1.0)
                + ((-0.5_f64).exp() / (2.0 * std::f64::consts::PI).sqrt()).ln(),
            (softplus_inverse(1.0_f64)? - 1.0)
                + ((-0.5_f64).exp() / (2.0 * std::f64::consts::PI).sqrt()).ln(),
            (softplus_inverse(1.0_f64)? - 1.0)
                + ((-2.0_f64).exp() / (2.0 * std::f64::consts::PI).sqrt()).ln(),
        ];

        for (computed, expected) in ess
            .state
            .get_ll_vector_view()
            .iter()
            .zip(expected_ll.iter())
        {
            assert_relative_eq!(computed, expected, epsilon = 1e-6);
        }

        let true_data_state = array![
            [0.0_f64, softplus_inverse(1.0_f64)?],
            [1.0_f64, softplus_inverse(1.0_f64)?],
            [-1.0_f64, softplus_inverse(1.0_f64)?],
            [2.0_f64, softplus_inverse(1.0_f64)?]
        ];
        assert_relative_eq!(
            true_data_state,
            ess.state.get_state_matrix_view(),
            epsilon = 1e-6
        );

        Ok(())
    }

    #[test]
    fn test_swap_wrapper() -> anyhow::Result<()> {
        let mut ess = EnsembleSliceSampler::new(2, 4, 2, None, None);

        ess.accept_proposed_state();

        Ok(())
    }
}
