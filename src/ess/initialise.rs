use std::time::Instant;

use ndarray::Array2;
use thiserror::Error;

use crate::{
    chains::ChainBuffer,
    ess::EnsembleSliceSampler,
    log_likelihood::{LogLikelihoodError, LogLikelihoodModel, WrappedLogLikelihoodModel},
};

#[derive(Debug, Error)]
pub enum InitialisationError {
    #[error("initialisation of initial state has failed: {0}")]
    InvalidLogLikelihood(#[from] LogLikelihoodError),
}

impl<C: ChainBuffer, L: LogLikelihoodModel> EnsembleSliceSampler<C, L> {
    pub fn initialise(&mut self, initial: &Array2<f64>) -> Result<(), InitialisationError> {
        self.state_i.get_mut_state_matrix().assign(
            &self
                .model
                .wrapped_columnar_transform_physical_to_internal(initial)?,
        );

        let start_time = Instant::now();
        for i in 0..self.n_walkers {
            *self.state_i.get_mut_ith_ll(i) = self
                .model
                .wrapped_log_likelihood(self.state_i.get_ith_state_vector(i))?
        }
        let duration = start_time.elapsed();

        println!(
            "INITIALISATION COMPLETE -- {} micro-s per walker",
            duration.as_micros() / self.n_walkers as u128
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use anyhow;
    use approx::assert_relative_eq;
    use ndarray::array;

    use crate::{functions::transforms::softplus_inverse, testing::helpers::build_test_ess};

    #[test]
    fn test_initialise() -> anyhow::Result<()> {
        let mut ess = build_test_ess();

        ess.initialise(&array![
            [0.0_f64, 1.0_f64],
            [1.0_f64, 1.0_f64],
            [-1.0_f64, 1.0_f64],
            [2.0_f64, 1.0_f64]
        ])?;

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
