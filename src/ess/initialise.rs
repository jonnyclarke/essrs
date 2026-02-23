use ndarray::Array2;

use crate::{
    FloatExt,
    chains::ChainBuffer,
    ess::{EnsemblSliceSampler, EnsembleSliceSamplerConfigTrait},
    log_likelihood::LogLikelihoodModel,
};

impl<
    T: FloatExt,
    CHAINS: ChainBuffer<T>,
    MODEL: LogLikelihoodModel<T>,
    CONFIG: EnsembleSliceSamplerConfigTrait,
> EnsemblSliceSampler<T, CHAINS, MODEL, CONFIG>
{
    pub fn initialise(&mut self, initial: &Array2<T>) {
        self.step.assign(initial);
        for i in 0..CONFIG::N_WALKERS {
            self.step_ll[i] = self
                .model
                .log_likelihood(&self.data, &self.step.row(i).to_owned())
        }
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
    use ndarray::array;

    use crate::testing::helpers::build_test_ess;

    #[test]
    fn test_initialise() {
        let mut ess = build_test_ess();

        ess.initialise(&array![
            [0.0_f32, 1.0_f32.ln()],
            [1.0_f32, 1.0_f32.ln()],
            [-1.0_f32, 1.0_f32.ln()],
            [2.0_f32, 1.0_f32.ln()],
            [-2.0_f32, 1.0_f32.ln()]
        ]);

        let expected_ll = array![
            (0.0_f32.exp() / (2.0 * std::f32::consts::PI).sqrt()).ln(),
            ((-0.5_f32).exp() / (2.0 * std::f32::consts::PI).sqrt()).ln(),
            ((-0.5_f32).exp() / (2.0 * std::f32::consts::PI).sqrt()).ln(),
            ((-2.0_f32).exp() / (2.0 * std::f32::consts::PI).sqrt()).ln(),
            ((-2.0_f32).exp() / (2.0 * std::f32::consts::PI).sqrt()).ln(),
        ];

        for (computed, expected) in ess.step_ll.iter().zip(expected_ll.iter()) {
            let tol = 1e-6_f32;
            println!("{} != {}", computed, expected);
            assert!((computed - expected).abs() < tol);
        }
    }
}
