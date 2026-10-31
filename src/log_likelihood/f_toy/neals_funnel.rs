use anyhow;
use ndarray::ArrayView1;

use crate::log_likelihood::{LogLikelihoodModel, ProbDataGivenThetaModel};

#[cfg_attr(all(doc, feature = "doc-math"), katexit::katexit)]
/// Likelihood construction of the 'Neals Funnel' toy problem.
///
/// This is a classic test case for MCMC algorithms from the bayesian hierarchical model.
/// This is a two dimensional problem, sampling x and y:
///
/// $$y \sim \mathcal{N}\left(0, 1\right)$$
/// $$x \sim \mathcal{N}\left(0, e^y\right)$$
///
/// The posterior probability is then:
/// $$
/// \mathcal{P}\left(x, y\right) \propto
/// \frac{1}{\sqrt{2\pi}}\exp{\left(-\frac{1}{2}y^2\right)} \cdot
/// \frac{1}{\sqrt{2\pi}e^y}\exp{\left(-\frac{1}{2}\left[\frac{x}{e^{y}}\right]^2\right)}
/// $$
///
/// and the log-likelihood is thus:
/// $$
/// \mathcal{L}\left(x, y\right) =  -\frac{1}{2} \left(
/// y^2 +\left[\frac{x}{e^{y}}\right]^2 + 2y \right) -
/// \ln\left(2\pi\right)
/// $$
pub struct AnalyticNealsFunnelToyLogL {}

impl AnalyticNealsFunnelToyLogL {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for AnalyticNealsFunnelToyLogL {
    fn default() -> Self {
        Self::new()
    }
}

impl LogLikelihoodModel for AnalyticNealsFunnelToyLogL {
    fn get_n_parameters(&self) -> usize {
        2
    }

    fn log_likelihood(&self, parameters: ArrayView1<f64>) -> anyhow::Result<f64> {
        let x = parameters[0];
        let y = parameters[1];

        let ll = -0.5 * (y.powi(2) + (x / y.exp()).powi(2) + 2.0 * y)
            - (2.0 * std::f64::consts::PI).ln();

        Ok(ll)
    }
}

impl ProbDataGivenThetaModel for AnalyticNealsFunnelToyLogL {}

#[cfg(test)]
mod tests {

    use ndarray::array;

    use super::*;

    #[test]
    fn test_initialisation() -> anyhow::Result<()> {
        let ll_neals_funnel = AnalyticNealsFunnelToyLogL::default();
        let parameters = array![1.0, 1.0];
        let _ = ll_neals_funnel.log_likelihood(parameters.view());

        Ok(())
    }
}
