use anyhow::ensure;
use ndarray::ArrayView1;

use crate::{
    functions::likelihood::log_normalised_gaussian,
    log_likelihood::{
        LogLikelihoodModel, ParameterTransformer, Parametrised, ToyModelSamplerComponent,
    },
};

/// One dimensional normal distribution
#[derive(Clone)]
pub struct AnalyticNormalToyLogL {
    mu: f64,
    sigma: f64,
}

impl AnalyticNormalToyLogL {
    /// Constructor
    ///
    /// # Arguments
    /// * `mu`: the mean of the normal distribution
    /// * `sigma`: the standard deviation of the normal distribution
    ///
    /// # Errors
    /// * passed 'sigma' value is <=0
    pub fn new(mu: f64, sigma: f64) -> anyhow::Result<Self> {
        ensure!(
            sigma.gt(&0.0_f64),
            "Sigma passed to toy normal distribution is invalid: {sigma}"
        );
        Ok(Self { mu, sigma })
    }
}

impl Parametrised for AnalyticNormalToyLogL {
    fn get_n_parameters(&self) -> usize {
        1
    }
}

impl ParameterTransformer for AnalyticNormalToyLogL {}

impl LogLikelihoodModel for AnalyticNormalToyLogL {
    fn log_likelihood(&self, parameters: ArrayView1<f64>) -> anyhow::Result<f64> {
        let x = parameters[0];

        let ll = log_normalised_gaussian(x, self.mu, self.sigma);

        Ok(ll)
    }
}

impl ToyModelSamplerComponent for AnalyticNormalToyLogL {}

#[cfg(test)]
mod tests {

    use ndarray::array;

    use super::*;

    #[test]
    fn test_initialisation() -> anyhow::Result<()> {
        let ll_normal = AnalyticNormalToyLogL::new(1.0, 3.0)?;
        let parameters = array![1.0];
        let _ = ll_normal.log_likelihood(parameters.view());

        Ok(())
    }
}
