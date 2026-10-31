
use anyhow;
use ndarray::ArrayView1;
use ndarray::Array1;

use crate::{functions::likelihood::log_normalised_gaussian, log_likelihood::LogLikelihoodModel};
use crate::functions::likelihood::log_normalised_gaussian_s2;
use crate::functions::numerical::log_sum_exp;
use crate::log_likelihood::ProbDataGivenThetaModel;

/// One dimensional normal distribution
#[derive(Clone)]
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

impl ProbDataGivenThetaModel for AnalyticNormalToyLogL {}

pub struct AnalyticNormalMixtureOneD {
    n: usize,

    theta: Array1<f64>,
    mu: Array1<f64>,
    sigma_sqr: Array1<f64>,
}

pub struct Missing;
pub struct Set;

/// One dimensional Gaussian mixture -- Builder
pub struct BuildAnalyticNormalMixtureOneD<T, M, S> {
    n: usize,

    theta: Option<Array1<f64>>,
    mu: Option<Array1<f64>>,
    sigma_sqr: Option<Array1<f64>>,

    _state: std::marker::PhantomData<(T, M, S)>,
}

impl BuildAnalyticNormalMixtureOneD<Missing, Missing, Missing> {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            theta: None,
            mu: None,
            sigma_sqr: None,
            _state: std::marker::PhantomData
        }
    }
}

impl<T, M, S> BuildAnalyticNormalMixtureOneD<T, M, S> {
    pub fn theta(self, theta: Array1<f64>) -> BuildAnalyticNormalMixtureOneD<Set, M, S> {
        let k = theta.len();
        assert_eq!(k, self.n-1);
        assert!(theta.sum().lt(&1.0));

        BuildAnalyticNormalMixtureOneD {
            n: self.n,
            theta: Some(theta),
            mu: self.mu,
            sigma_sqr: self.sigma_sqr,
            _state: std::marker::PhantomData
        }
    }

    pub fn mu(self, mu: Array1<f64>) -> BuildAnalyticNormalMixtureOneD<M, Set, S> {
        let k = mu.len();
        assert_eq!(k, self.n);

        BuildAnalyticNormalMixtureOneD {
            n: self.n,
            theta: self.theta,
            mu: Some(mu),
            sigma_sqr: self.sigma_sqr,
            _state: std::marker::PhantomData
        }
    }

    pub fn sigma(self, sigma: Array1<f64>) -> BuildAnalyticNormalMixtureOneD<M, M, Set> {
        let k = sigma.len();
        assert_eq!(k, self.n);

        BuildAnalyticNormalMixtureOneD {
            n: self.n,
            theta: self.theta,
            mu: self.mu,
            sigma_sqr: Some(sigma.into_iter().map(|x| x.powi(2)).collect()),
            _state: std::marker::PhantomData
        }
    }
}

impl BuildAnalyticNormalMixtureOneD<Set, Set, Set> {
    pub fn build(self) -> AnalyticNormalMixtureOneD{
        AnalyticNormalMixtureOneD {
            n: self.n,
            theta: self.theta.unwrap(),
            mu: self.mu.unwrap(),
            sigma_sqr: self.sigma_sqr.unwrap(),
        }
    }
}

impl LogLikelihoodModel for AnalyticNormalMixtureOneD {
    fn get_n_parameters(&self) -> usize {
        1
    }

    fn log_likelihood(&self, parameters: ArrayView1<f64>) -> anyhow::Result<f64> {
        let x = parameters[0];

        let mut ll = self.theta[0].ln() + log_normalised_gaussian_s2(x, self.mu[0], self.sigma_sqr[0]);

        for i in 1..(self.n - 1) {
            ll = log_sum_exp(
                ll,
                self.theta[i].ln() + log_normalised_gaussian_s2(x, self.mu[i], self.sigma_sqr[i])
            )?
        }

        ll = log_sum_exp(
            ll,
            (1.0 - self.theta.sum()).ln() + log_normalised_gaussian_s2(x, self.mu[self.n-1], self.sigma_sqr[self.n-1])
        )?;

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
