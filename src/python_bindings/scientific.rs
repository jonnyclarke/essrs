use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::sync::Arc;

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

    let model = Arc::new(GaussianLl1dDataErrors::from_mu_sigma(&mu, &sigma));

    PyLogLikelihoodModel {
        model: model.clone(),
        p_data_g_theta: None,
        p_theta_g_data: Some(model),
    }
}

#[pyfunction]
pub fn normal_2d_gmm_deconv(
    n_components: usize,
    data: PyReadonlyArray2<f64>,
) -> PyLogLikelihoodModel {
    let d = data.as_array().to_owned();
    PyLogLikelihoodModel {
        model: Arc::new(GaussianMixModel2DimUnDeConv::new(d, n_components)),
        p_data_g_theta: None,
        p_theta_g_data: None,
    }
}
