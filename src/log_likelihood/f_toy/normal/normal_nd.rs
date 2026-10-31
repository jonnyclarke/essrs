use anyhow::{anyhow, bail};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, ArrayView1};

use crate::{
    functions::{
        helper::{array1_to_dvector, arrayview1_to_dvector, covariance_to_dmatrix},
        likelihood::normal_n_dimensional::{
            cholesky_l_to_log_det, covariance_to_cholesky_l, mahalanobis_from_cholesky,
            ndim_to_norm_const, normal_nd_ll,
        },
    },
    log_likelihood::LogLikelihoodModel,
};

pub struct AnalyticNormalNDimensional {
    n_dimensions: usize,

    mu: DVector<f64>,

    /// normalisation constant
    ln_n_pi_const: f64,
    log_det: f64,
    cholesky_l: DMatrix<f64>,
}

pub struct Missing;
pub struct Set;

/// One dimensional Gaussian mixture -- Builder
pub struct BuildAnalyticNormalNDimensional<M, S, C> {
    n_dimensions: usize,

    mu: Option<Array1<f64>>,
    sigma: Option<Array1<f64>>,
    correlation: Option<Array1<f64>>,
    errors: Vec<anyhow::Error>,

    _state: std::marker::PhantomData<(M, S, C)>,
}

impl BuildAnalyticNormalNDimensional<Missing, Missing, Missing> {
    pub fn new(n_dimensions: usize) -> Self {
        Self {
            n_dimensions,
            mu: None,
            sigma: None,
            correlation: None,
            errors: vec![],
            _state: std::marker::PhantomData,
        }
    }
}

impl<M, S, C> BuildAnalyticNormalNDimensional<M, S, C> {
    pub fn mu(mut self, mu: Array1<f64>) -> BuildAnalyticNormalNDimensional<Set, S, C> {
        if mu.len() != self.n_dimensions {
            self.errors.push(anyhow!(
                "mu length, ({}), does not match dimensionallity ({})",
                mu.len(),
                self.n_dimensions
            ));
        }

        BuildAnalyticNormalNDimensional {
            n_dimensions: self.n_dimensions,
            mu: Some(mu),
            sigma: self.sigma,
            correlation: self.correlation,
            errors: self.errors,
            _state: std::marker::PhantomData,
        }
    }

    pub fn sigma(mut self, sigma: Array1<f64>) -> BuildAnalyticNormalNDimensional<M, Set, C> {
        if sigma.len() != self.n_dimensions {
            self.errors.push(anyhow!(
                "sigma length, ({}), does not match dimensionallity ({})",
                sigma.len(),
                self.n_dimensions
            ));
        }

        BuildAnalyticNormalNDimensional {
            n_dimensions: self.n_dimensions,
            mu: self.mu,
            sigma: Some(sigma),
            correlation: self.correlation,
            errors: self.errors,
            _state: std::marker::PhantomData,
        }
    }

    pub fn correlation(
        mut self,
        correlation: Array1<f64>,
    ) -> BuildAnalyticNormalNDimensional<M, S, Set> {
        let expected = self.n_dimensions * (self.n_dimensions - 1) / 2;
        if correlation.len().ne(&expected) {
            self.errors.push(
                anyhow!(
                    "sigma length, ({}), does not match dimensionallity expectation [{} = n * (n - 1) / 2]",
                    correlation.len(),
                    expected
                )
            );
        }

        BuildAnalyticNormalNDimensional {
            n_dimensions: self.n_dimensions,
            mu: self.mu,
            sigma: self.sigma,
            correlation: Some(correlation),
            errors: self.errors,
            _state: std::marker::PhantomData,
        }
    }
}

impl BuildAnalyticNormalNDimensional<Set, Set, Set> {
    pub fn build(self) -> anyhow::Result<AnalyticNormalNDimensional> {
        if !self.errors.is_empty() {
            bail!(
                "Builder validation failed:\n{}",
                self.errors
                    .iter()
                    .map(|e| format!("- {e}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }

        let cov = covariance_to_dmatrix(&self.sigma.unwrap(), &self.correlation.unwrap()).unwrap();
        let l = covariance_to_cholesky_l(&cov);

        Ok(AnalyticNormalNDimensional {
            n_dimensions: self.n_dimensions,

            mu: array1_to_dvector(&self.mu.unwrap()),

            ln_n_pi_const: ndim_to_norm_const(self.n_dimensions),
            log_det: cholesky_l_to_log_det(&l),
            cholesky_l: l,
        })
    }
}

impl LogLikelihoodModel for AnalyticNormalNDimensional {
    fn get_n_parameters(&self) -> usize {
        self.n_dimensions
    }

    fn log_likelihood(&self, parameters: ArrayView1<f64>) -> anyhow::Result<f64> {
        let matmul = mahalanobis_from_cholesky(
            &arrayview1_to_dvector(&parameters),
            &self.mu,
            &self.cholesky_l,
        );
        let ll = normal_nd_ll(self.ln_n_pi_const, self.log_det, matmul);

        Ok(ll)
    }
}

#[cfg(test)]
mod tests {

    use ndarray::array;
    use rstest::rstest;

    use super::*;
    // use crate::functions::likelihood::log_normalised_gaussian;
    use crate::functions::likelihood::log_normal_2d;

    #[rstest]
    #[case(
            2,
            array![0.5],
            array![0.5, 1.0],
            array![0.4],
        )
    ]
    #[case(
            2,
            array![0.5, 0.3],
            array![0.5, 1.0, 2.0],
            array![0.4, 1.0],
        )
    ]
    fn test_build_failures(
        #[case] n_dimensions: usize,
        #[case] mu: Array1<f64>,
        #[case] sigma: Array1<f64>,
        #[case] correlation: Array1<f64>,
    ) -> anyhow::Result<()> {
        let result = BuildAnalyticNormalNDimensional::new(n_dimensions)
            .mu(mu)
            .sigma(sigma)
            .correlation(correlation)
            .build();

        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_get_n_parameters() -> anyhow::Result<()> {
        let model = BuildAnalyticNormalNDimensional::new(2)
            .mu(array![0.0, 0.0])
            .sigma(array![0.5, 0.9])
            .correlation(array![0.4])
            .build()?;

        let _ = model.get_n_parameters();

        Ok(())
    }

    #[rstest]
    #[case(0.0, 1.0)]
    fn test_log_likelihood(#[case] x: f64, #[case] y: f64) -> anyhow::Result<()> {
        let model = BuildAnalyticNormalNDimensional::new(2)
            .mu(array![0.0, 0.0])
            .sigma(array![0.5, 0.9])
            .correlation(array![0.4])
            .build()?;

        assert_eq!(
            model.log_likelihood(array![x, y].view())?,
            log_normal_2d(
                x,
                y,
                0.0,
                0.0,
                0.5_f64.powi(2),
                0.4 * 0.5 * 0.9,
                0.9_f64.powi(2)
            )
        );
        Ok(())
    }
}
