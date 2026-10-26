use ndarray::{Array3, ArrayView2};

use crate::{
    chains::ChainBuffer,
    ess::InitialisationError,
    log_likelihood::{LogLikelihoodModel, WrappedLogLikelihoodModel},
    moves::EnsembleMove,
    state::WalkerState,
};

/// Ensemble Slice Sampler Object.
///
/// This struct manages all the internals of the MCMC fit.
/// Type is generic over:
/// * implementor of 'ChainBuffer' trait: used to store MCMC history
/// * implementor of 'MoveHandler' trait: controls how the MCMC makes proposals
/// * implementor of 'LogLikelihoodModel' trait: provides log-likelihood computation
pub struct EnsembleSliceSampler<C: ChainBuffer, M: EnsembleMove, L: LogLikelihoodModel> {
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

impl<C: ChainBuffer, M: EnsembleMove, L: LogLikelihoodModel> EnsembleSliceSampler<C, M, L> {
    pub fn new(
        max_n_steps: usize,
        n_walkers: usize,
        n_parameters: usize,
        chains: C,
        move_handler: M,
        model: L,
    ) -> Self {
        let required_walkers = (n_parameters * 2).max(3);
        assert!(n_walkers >= required_walkers);

        Self {
            max_n_steps,
            n_walkers,
            n_parameters,

            chains,
            move_handler,
            model,

            state: WalkerState::new(n_walkers, n_parameters),
        }
    }

    pub fn initialise(&mut self, physical: ArrayView2<f64>) -> Result<(), InitialisationError> {
        let mut internal = physical.to_owned();

        self.model
            .wrapped_columnar_transform_physical_to_internal(physical, internal.view_mut())?;

        // this is used simply to pass to log-likelihood in case we need to report an error
        // TODO: can get rid of this allocation?
        let mut physical = self.state.get_ith_state_vector_view(0).to_owned();

        for i in 0..self.n_walkers {
            self.state.assign_ith_ll(
                i,
                self.model
                    .wrapped_log_likelihood(internal.row(i), &mut physical.view_mut())?,
            )
        }

        self.state.assign_state_matrix(internal.view());

        // we have been working with next. swap next -> current
        self.state.swap();

        Ok(())
    }

    /// Calls state-buffer swap which flips current and next memory locations
    pub fn accept_proposed_state(&mut self) {
        self.state.swap();
    }

    /// Run burn in phase to converge walkers to posterior probability region
    pub fn run_burn_in<T: EnsembleMove>(
        &mut self,
        n_burn_in: usize,
        burn_move_handler: &T,
    ) -> anyhow::Result<()> {
        for iteration in 1..=n_burn_in {
            burn_move_handler.jump(iteration, &self.model, &mut self.state)?;
            self.accept_proposed_state(); // we accept new state before storing to avoid storing the initial conditions state...
        }

        Ok(())
    }

    /// Run the MCMC sampler
    ///
    /// # Arguments
    /// * n_burn_in: the number of iterations to allow walker convergence prior to storing samples
    pub fn run_sampler(&mut self) -> anyhow::Result<()> {
        for iteration in 1..=self.max_n_steps {
            self.move_handler
                .jump(iteration, &self.model, &mut self.state)?;
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

    /// Extract full state
    ///
    /// This wraps the chain buffer object's 'extract_state' function
    pub fn extract_state(&self) -> Array3<f64> {
        self.chains.extract_state()
    }
}

#[cfg(test)]
mod tests_ess {
    use anyhow;
    use approx::assert_relative_eq;
    use ndarray::array;

    use crate::{
        functions::transforms::softplus::softplus_inverse, testing::helpers::build_test_ess,
    };

    #[test]
    fn test_initialise() -> anyhow::Result<()> {
        let mut ess = build_test_ess();

        let data = array![
            [0.0_f64, 1.0_f64],
            [1.0_f64, 1.0_f64],
            [-1.0_f64, 1.0_f64],
            [2.0_f64, 1.0_f64]
        ];
        ess.initialise(data.view())?;

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
}
