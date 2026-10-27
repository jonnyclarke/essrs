use anyhow;
use ndarray::ArrayView1;

use crate::{functions::likelihood::log_normalised_gaussian, log_likelihood::LogLikelihoodModel};

pub struct AnalyticNormalToyLogL {
    mu: f64,
    sigma: f64,
}

impl AnalyticNormalToyLogL {
    pub fn new(mu: f64, sigma: f64) -> Self {
        Self { mu, sigma }
    }
}

impl LogLikelihoodModel for AnalyticNormalToyLogL {
    fn get_n_parameters(&self) -> usize {
        1
    }

    fn log_likelihood(&self, parameters: ArrayView1<f64>) -> anyhow::Result<f64> {
        let x = parameters[0];

        let ll = log_normalised_gaussian(x, self.mu, self.sigma);

        Ok(ll)
    }
}

#[cfg(test)]
mod tests {

    use ndarray::array;

    use super::*;

    #[test]
    fn test_initialisation() -> anyhow::Result<()> {
        let ll_normal = AnalyticNormalToyLogL::new(1.0, 3.0);
        let parameters = array![1.0];
        let _ = ll_normal.log_likelihood(parameters.view());

        Ok(())
    }
}
