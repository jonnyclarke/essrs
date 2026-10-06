use ndarray::ArrayView2;
use thiserror::Error;

use crate::{
    chains::ChainBuffer,
    ess::EnsembleSliceSampler,
    log_likelihood::{LogLikelihoodError, WrappedLogLikelihoodModel},
    moves::EnsembleMoveHandler,
};

#[derive(Debug, Error)]
pub enum InitialisationError {
    #[error("initialisation of initial state has failed: {0}")]
    InvalidLogLikelihood(#[from] LogLikelihoodError),
}

impl<C: ChainBuffer, M: EnsembleMoveHandler, L: WrappedLogLikelihoodModel>
    EnsembleSliceSampler<C, M, L>
{
    pub fn initialise(&mut self, physical: ArrayView2<f64>) -> Result<(), InitialisationError> {
        let internal = self.state_i.get_mut_state_matrix();

        self.model
            .wrapped_columnar_transform_physical_to_internal(physical, internal)?;

        // this is used simply to pass to log-likelihood in case we need to report an error
        // TODO: can get rid of this allocation?
        let mut physical = self.state_i.get_ith_state_vector(0).to_owned();

        for i in 0..self.n_walkers {
            *self.state_i.get_mut_ith_ll(i) = self.model.wrapped_log_likelihood(
                self.state_i.get_ith_state_vector(i),
                &mut physical.view_mut(),
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
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

        for (computed, expected) in ess.state_i.get_ll_vector().iter().zip(expected_ll.iter()) {
            assert_relative_eq!(computed, expected, epsilon = 1e-6);
        }

        Ok(())
    }
}
