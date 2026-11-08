use std::sync::Arc;

use anyhow::Context;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::{
    log_likelihood::f_scientific::{
        GaussianLl1dDataErrors, GaussianMixModel2DimUnDeConv, Normal1DimMixErrDeconv,
    },
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
        sampler_component: None,
    }
}

#[pyfunction]
pub fn normal_1d_mix_deconv(
    mu: PyReadonlyArray1<f64>,
    sigma: PyReadonlyArray1<f64>,
    n: usize,
) -> PyResult<PyLogLikelihoodModel> {
    let mu = mu.as_array().to_owned();
    let sigma = sigma.as_array().to_owned();

    let model = Arc::new(Normal1DimMixErrDeconv::new(mu, sigma, n)?);

    Ok(PyLogLikelihoodModel {
        model: model.clone(),
        sampler_component: None,
    })
}

#[pyfunction]
pub fn normal_2d_gmm_deconv(
    n_components: usize,
    data: PyReadonlyArray2<f64>,
) -> PyResult<PyLogLikelihoodModel> {
    let d = data.as_array().to_owned();
    Ok(PyLogLikelihoodModel {
        model: Arc::new(
            GaussianMixModel2DimUnDeConv::new(d, n_components)
                .context("construction of normal-2d python binding has failed")?,
        ),
        sampler_component: None,
    })
}
