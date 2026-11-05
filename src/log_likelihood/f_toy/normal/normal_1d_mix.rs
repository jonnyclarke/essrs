use anyhow::{anyhow, ensure};
use ndarray::{Array1, ArrayView1};

use crate::{
    functions::{likelihood::log_normalised_gaussian_s2, numerical::log_sum_exp},
    log_likelihood::{LogLikelihoodModel, ParameterTransformer, Parametrised},
};

/// Normal 1D sampler mixture model. P(data|theta).
///
/// # Fields:
/// * n: the number of normal distributions in the mixture model
///
/// ...
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
    errors: Vec<anyhow::Error>,

    _state: std::marker::PhantomData<(T, M, S)>,
}

impl BuildAnalyticNormalMixtureOneD<Missing, Missing, Missing> {
    /// Constructor
    ///
    /// # Arguments
    /// * n: the number of one-dimensional gaussian distributions contained in the mixture.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            theta: None,
            mu: None,
            sigma_sqr: None,
            errors: vec![],

            _state: std::marker::PhantomData,
        }
    }
}

impl<T, M, S> BuildAnalyticNormalMixtureOneD<T, M, S> {
    /// Setter for 'theta' value
    ///
    /// # Arguments
    /// * `theta`: Array of weights for each of the individual components
    ///
    /// # Errors
    /// * returns an error if the length of `theta` is not equal to (n - 1); the weight of the final component is equal to (1 - `theta.sum()`)
    /// * returns an error if any of the `theta` values are <= 0
    /// * returns an error if the sum of the `theta` array is greater than 1; represents an ill defined mixture integral
    pub fn theta(mut self, theta: Array1<f64>) -> BuildAnalyticNormalMixtureOneD<Set, M, S> {
        if theta.len() != self.n - 1 {
            self.errors.push(anyhow!(
                "theta length, k={}, inconsistent with number of components, n={}, require k = n-1",
                theta.len(),
                self.n
            ));
        }

        if theta.iter().any(|x| x.le(&0.0_f64)) {
            self.errors.push(anyhow!(
                "invalid theta value, theta cannot be negative:\n\t{:#?}",
                theta
            ))
        }

        if theta.sum().ge(&1.0_f64) {
            self.errors
                .push(anyhow!("theta sum is invalid :: {} > 1", theta.sum()))
        }

        BuildAnalyticNormalMixtureOneD {
            n: self.n,
            theta: Some(theta),
            mu: self.mu,
            sigma_sqr: self.sigma_sqr,
            errors: self.errors,
            _state: std::marker::PhantomData,
        }
    }

    /// Setter for 'mu' field
    ///
    /// # Arguments:
    /// * mu: Array containing normal distribution centroid values
    ///
    /// # Errors:
    /// * Returns an error if the length of 'mu' deos not equal the specified number of components.
    pub fn mu(mut self, mu: Array1<f64>) -> BuildAnalyticNormalMixtureOneD<M, Set, S> {
        if mu.len() != self.n {
            self.errors.push(anyhow!(
                "mean length, k={}, inconsistent with number of components, n={}, require k = n",
                mu.len(),
                self.n
            ));
        }

        BuildAnalyticNormalMixtureOneD {
            n: self.n,
            theta: self.theta,
            mu: Some(mu),
            sigma_sqr: self.sigma_sqr,
            errors: self.errors,
            _state: std::marker::PhantomData,
        }
    }

    /// Setter for 'sigma' field
    ///
    /// # Arguments:
    /// * sigma: Array containing normal distribution standard deviation values
    ///
    /// # Errors:
    /// * returns an error if the number of sigma values does not match the specified number of components
    /// * returns an error if any of the given sigma values are <= 0
    pub fn sigma(mut self, sigma: Array1<f64>) -> BuildAnalyticNormalMixtureOneD<M, M, Set> {
        if sigma.len() != self.n {
            self.errors.push(anyhow!(
                "sigma length, k={}, inconsistent with number of components, n={}, require k = n",
                sigma.len(),
                self.n
            ));
        }

        if sigma.iter().any(|s| s.le(&0.0_f64)) {
            self.errors
                .push(anyhow!("negative sigma is invalid::\n\t{sigma:#?}"));
        }

        BuildAnalyticNormalMixtureOneD {
            n: self.n,
            theta: self.theta,
            mu: self.mu,
            sigma_sqr: Some(sigma.into_iter().map(|x| x.powi(2)).collect()),
            errors: self.errors,
            _state: std::marker::PhantomData,
        }
    }
}

impl BuildAnalyticNormalMixtureOneD<Set, Set, Set> {
    /// Builder converting builder struct to model
    ///
    /// # Errors
    /// * returns an error if any errors during the build phase have accumulated
    ///
    /// # Panics
    /// * `theta` is still `None`
    /// * `mu` is still `None`
    /// * `sigma_sqr` is still '`None`
    pub fn build(self) -> anyhow::Result<AnalyticNormalMixtureOneD> {
        ensure!(
            self.errors.is_empty(),
            "Builder validations have failed:\n{}",
            self.errors
                .iter()
                .map(|e| format!("- {e}"))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(AnalyticNormalMixtureOneD {
            n: self.n,
            theta: self.theta.unwrap(),
            mu: self.mu.unwrap(),
            sigma_sqr: self.sigma_sqr.unwrap(),
        })
    }
}

impl Parametrised for AnalyticNormalMixtureOneD {
    fn get_n_parameters(&self) -> usize {
        1
    }
}

impl ParameterTransformer for AnalyticNormalMixtureOneD {}

impl LogLikelihoodModel for AnalyticNormalMixtureOneD {
    fn log_likelihood(&self, parameters: ArrayView1<f64>) -> anyhow::Result<f64> {
        let x = parameters[0];

        let mut ll = (1.0_f64 - self.theta.sum()).ln()
            + log_normalised_gaussian_s2(x, self.mu[self.n - 1], self.sigma_sqr[self.n - 1]);

        for i in 0..(self.n - 1) {
            ll = log_sum_exp(
                ll,
                self.theta[i].ln() + log_normalised_gaussian_s2(x, self.mu[i], self.sigma_sqr[i]),
            )?
        }

        Ok(ll)
    }
}

#[cfg(test)]
mod tests {

    use ndarray::array;
    use rstest::rstest;

    use super::*;
    use crate::functions::likelihood::log_normalised_gaussian;

    #[rstest]
    #[case(
            3,
            array![0.5],
            array![0.0, 1.0],
            array![1.0, 1.0],
        )
    ]
    #[case(
            2,
            array![0.5],
            array![0.0, 1.0],
            array![1.0, 1.0, 2.0],
        )
    ]
    #[case(
            3,
            array![0.5, 0.6],
            array![0.0, 1.0, 2.0],
            array![1.0, 1.0, 0.01],
        )
    ]
    #[case(
            3,
            array![0.5, 0.3],
            array![0.0, 1.0, 2.0],
            array![1.0, 1.0, 0.0],
        )
    ]
    #[case(
            3,
            array![0.5, 0.3],
            array![0.0, 1.0, 2.0],
            array![1.0, 1.0, -1.0],
        )
    ]
    fn test_build_failures(
        #[case] n: usize,
        #[case] theta: Array1<f64>,
        #[case] mu: Array1<f64>,
        #[case] sigma: Array1<f64>,
    ) -> anyhow::Result<()> {
        let builder = BuildAnalyticNormalMixtureOneD::new(n);

        let result = builder.theta(theta).mu(mu).sigma(sigma).build();

        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_log_likelihood() -> anyhow::Result<()> {
        let model = BuildAnalyticNormalMixtureOneD::new(3)
            .theta(array![0.4, 0.3])
            .mu(array![-1.0, 0.0, 1.0])
            .sigma(array![1.0, 2.0, 3.0])
            .build()?;

        assert_eq!(model.get_n_parameters(), 1);

        assert_eq!(
            model.log_likelihood(array![0.5].view())?,
            log_sum_exp(
                0.4_f64.ln() + log_normalised_gaussian(0.5, -1.0, 1.0),
                log_sum_exp(
                    0.3_f64.ln() + log_normalised_gaussian(0.5, 0.0, 2.0),
                    0.3_f64.ln() + log_normalised_gaussian(0.5, 1.0, 3.0)
                )?
            )?
        );

        Ok(())
    }

    // #[test]
    // fn test_init_failure_theta_too_large() -> anyhow::Result<()> {
    //     let result = DynToyMixModelDistributionSampler::new(
    //         array![1.5],
    //         vec![
    //             Arc::new(AnalyticNormalToyLogL::new(0.0, 1.0)),
    //             Arc::new(AnalyticNormalToyLogL::new(1.0, 2.0)),
    //         ]
    //     );
    //     assert!(result.is_err());

    //     Ok(())
    // }

    // #[rstest]
    // #[case(array![0.2])]
    // #[case(array![0.2, 0.1, 0.1])]
    // fn test_init_failure_theta_wrong_length(#[case] theta: Array1<f64>) -> anyhow::Result<()> {
    //     let result = DynToyMixModelDistributionSampler::new(
    //         theta,
    //         vec![
    //             Arc::new(AnalyticNormalToyLogL::new(0.0, 1.0)),
    //             Arc::new(AnalyticNormalToyLogL::new(1.0, 2.0)),
    //             Arc::new(AnalyticNormalToyLogL::new(1.0, 2.0)),
    //         ]
    //     );
    //     assert!(result.is_err());

    //     Ok(())
    // }
}
