//! Helper functions to directly modify log-likelihood to apply prior distributions on given parameters.
//!
use crate::functions::likelihood::log_normalised_gaussian;

#[cfg_attr(all(doc, feature = "doc-math"), katexit::katexit)]
/// Exponential prior
///
/// $$
/// P(x) = \lambda * e^{-\lambda \cdot x}
/// $$
/// $$
/// \ln\left( P\left(x\right)\right) = \ln(\left(\lambda\right) - \lambda \cdot x
/// $$
/// where lambda controls the scale of the distribution.
///
/// NOTE: this function has no return value. Instead it modifies the passed reference to the log-likelihood.
///
/// # Arguments
/// * `x` - parameter for which prior is to be applied.
/// * `lambda` - the scale of the distribution.
/// * `&mut ll` - reference to mutable log-likelihood vartiable.
pub fn ln_prior_exponential(x: f64, lambda: f64, ll: &mut f64) {
    *ll += lambda.ln() - lambda * x
}

pub fn ln_prior_normal(x: f64, mu: f64, sigma: f64, ll: &mut f64) {
    *ll += log_normalised_gaussian(x, mu, sigma)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(0.0, 1.0, 1.0_f64.ln())]
    #[case(1.0, 1.0, (1.0_f64 * (-1.0_f64).exp()).ln())]
    #[case(1.234, 2.345, (2.345_f64 * (-2.345_f64 * 1.234_f64).exp()).ln())]
    fn test_ln_prior_exponential(#[case] x: f64, #[case] lambda: f64, #[case] target: f64) {
        let mut ll: f64 = 0.0;
        ln_prior_exponential(x, lambda, &mut ll);
        assert_eq!(ll, target);
    }

    #[rstest]
    #[case(
        0.0, 1.0, 2.0,
        -0.5 * 0.5_f64.powi(2) - 0.5 * (2.0 * std::f64::consts::PI * 2.0_f64.powi(2)).ln()
    )]
    fn test_ln_prior_normal(
        #[case] x: f64,
        #[case] mu: f64,
        #[case] sigma: f64,
        #[case] target: f64,
    ) {
        let mut ll: f64 = 0.0;
        ln_prior_normal(x, mu, sigma, &mut ll);
        assert_eq!(ll, target);
    }
}
