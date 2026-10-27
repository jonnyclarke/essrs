use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::{
    log_likelihood::f_scientific::{GaussianLl1dDataErrors, GaussianMixModel2DimUnDeConv},
    python_bindings::PyLogLikelihoodModel,
};

#[pyfunction]
pub fn normal_1d_deconv(
    mu: PyReadonlyArray1<f64>,
    sigma: PyReadonlyArray1<f64>,
) -> PyLogLikelihoodModel {
    let mu = mu.as_array().to_owned();
    let sigma = sigma.as_array().to_owned();
    PyLogLikelihoodModel {
        inner: Box::new(GaussianLl1dDataErrors::from_mu_sigma(&mu, &sigma)),
    }
}

#[pyfunction]
pub fn normal_2d_gmm_deconv(
    n_components: usize,
    data: PyReadonlyArray2<f64>,
) -> PyLogLikelihoodModel {
    let d = data.as_array().to_owned();
    PyLogLikelihoodModel {
        inner: Box::new(GaussianMixModel2DimUnDeConv::new(d, n_components)),
    }
}
