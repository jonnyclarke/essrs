use std::time::Instant;

use ndarray::Array2;

use crate::{
    chains::ChainBuffer,
    ess::{EnsemblSliceSampler, EnsembleSliceSamplerConfigTrait},
    log_likelihood::LogLikelihoodModel,
};

impl<CHAINS: ChainBuffer, MODEL: LogLikelihoodModel, CONFIG: EnsembleSliceSamplerConfigTrait>
    EnsemblSliceSampler<CHAINS, MODEL, CONFIG>
{
    pub fn initialise(&mut self, initial: &Array2<f64>) {
        self.state_i.get_mut_state_matrix().assign(initial);
        let start_time = Instant::now();
        for i in 0..CONFIG::N_WALKERS {
            *self.state_i.get_mut_ith_ll(i) = self
                .model
                .log_likelihood(self.state_i.get_ith_state_vector(i))
        }
        let duration = start_time.elapsed();

        println!(
            "INITIALISATION COMPLETE -- {} micro-s per walker",
            duration.as_micros() / CONFIG::N_WALKERS as u128
        );
        self.print_quantile_summary();

        self.state_i.panic_on_invalid_ll();
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::array;

    use crate::{functions::transforms::inv_softplus, testing::helpers::build_test_ess};

    #[test]
    fn test_initialise() {
        let mut ess = build_test_ess();

        ess.initialise(&array![
            [0.0_f64, inv_softplus(1.0_f64)],
            [1.0_f64, inv_softplus(1.0_f64)],
            [-1.0_f64, inv_softplus(1.0_f64)],
            [2.0_f64, inv_softplus(1.0_f64)]
        ]);

        let expected_ll = array![
            (inv_softplus(1.0_f64) - 1.0)
                + (0.0_f64.exp() / (2.0 * std::f64::consts::PI).sqrt()).ln(),
            (inv_softplus(1.0_f64) - 1.0)
                + ((-0.5_f64).exp() / (2.0 * std::f64::consts::PI).sqrt()).ln(),
            (inv_softplus(1.0_f64) - 1.0)
                + ((-0.5_f64).exp() / (2.0 * std::f64::consts::PI).sqrt()).ln(),
            (inv_softplus(1.0_f64) - 1.0)
                + ((-2.0_f64).exp() / (2.0 * std::f64::consts::PI).sqrt()).ln(),
        ];

        for (computed, expected) in ess.state_i.get_ll_vector().iter().zip(expected_ll.iter()) {
            assert_relative_eq!(computed, expected, epsilon = 1e-6);
        }
    }
}
