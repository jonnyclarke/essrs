
use ndarray::{Array1, ArrayView1};
use nalgebra::{DVector, DMatrix};

use crate::log_likelihood::LogLikelihoodModel;
use crate::functions::helper::{array1_to_dvector, arrayview1_to_dvector};
use crate::functions::likelihood::ndim_to_norm_const;
use crate::functions::likelihood::covariance_to_cholesky_l;
use crate::functions::helper::covariance_to_dmatrix;
use crate::functions::likelihood::cholesky_l_to_log_det;
use crate::functions::likelihood::mahalanobis_from_cholesky;
use crate::functions::likelihood::normal_nd_ll;


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

    _state: std::marker::PhantomData<(M, S, C)>,
}

impl BuildAnalyticNormalNDimensional<Missing, Missing, Missing> {
    pub fn new(n_dimensions: usize) -> Self {
        Self {
            n_dimensions,
            mu: None,
            sigma: None,
            correlation: None,
            _state: std::marker::PhantomData
        }
    }
}

impl<M, S, C> BuildAnalyticNormalNDimensional<M, S, C> {

    pub fn mu(self, mu: Array1<f64>) -> BuildAnalyticNormalNDimensional<Set, S, C> {
        let k = mu.len();
        assert_eq!(k, self.n_dimensions);

        BuildAnalyticNormalNDimensional {
            n_dimensions: self.n_dimensions,
            mu: Some(mu),
            sigma: self.sigma,
            correlation: self.correlation,
            _state: std::marker::PhantomData
        }
    }

    pub fn sigma(self, sigma: Array1<f64>) -> BuildAnalyticNormalNDimensional<M, Set, C> {
        let k = sigma.len();
        assert_eq!(k, self.n_dimensions);

        BuildAnalyticNormalNDimensional {
            n_dimensions: self.n_dimensions,
            mu: self.mu,
            sigma: Some(sigma),
            correlation: self.correlation,
            _state: std::marker::PhantomData
        }
    }

    pub fn correlation(self, correlation: Array1<f64>) -> BuildAnalyticNormalNDimensional<M, S, Missing> {
        let k = correlation.len();
        assert_eq!(k, self.n_dimensions * (self.n_dimensions - 1) / 2);

        BuildAnalyticNormalNDimensional {
            n_dimensions: self.n_dimensions,
            mu: self.mu,
            sigma: self.sigma,
            correlation: Some(correlation),
            _state: std::marker::PhantomData
        }
    }
}

impl BuildAnalyticNormalNDimensional<Set, Set, Set> {
    pub fn build(self) -> AnalyticNormalNDimensional{

        let cov = covariance_to_dmatrix(
            &self.sigma.unwrap(),
            &self.correlation.unwrap()
        ).unwrap();
        let l = covariance_to_cholesky_l(&cov);

        AnalyticNormalNDimensional {
            n_dimensions: self.n_dimensions,

            mu: array1_to_dvector(&self.mu.unwrap()),

            ln_n_pi_const: ndim_to_norm_const(self.n_dimensions),
            log_det: cholesky_l_to_log_det(&l),
            cholesky_l: l,
        }
    }
}

impl LogLikelihoodModel for AnalyticNormalNDimensional {
    fn get_n_parameters(&self) -> usize {
        self.n_dimensions
    }

    fn log_likelihood(&self, parameters: ArrayView1<f64>) -> anyhow::Result<f64> {

        let ll = normal_nd_ll(
            self.ln_n_pi_const,
            self.log_det,
            mahalanobis_from_cholesky(
                &arrayview1_to_dvector(&parameters),
                &self.mu,
                &self.cholesky_l
            )
        );

        Ok(ll)
    }
}
