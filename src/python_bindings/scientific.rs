use std::sync::Arc;

use anyhow::Context;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::{
    log_likelihood::f_scientific::{GaussianLl1dDataErrors, GaussianMixModel2DimUnDeConv},
    python_bindings::PyLogLikelihoodModel,
    log_likelihood::f_scientific::DynNcompMixtureModel,
};

#[pyfunction]
pub fn dyn_ftheta_ncomp_mm(
    list_models: Vec<PyRef<'_, PyLogLikelihoodModel>>,
) -> PyResult<PyLogLikelihoodModel> {
    let components = list_models
        .iter()
        .map(|model| {
            model
                .p_theta_g_data
                .as_ref()
                .cloned()
                .context("Model does not have a distribution sampler")
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let model = DynNcompMixtureModel::new(components).expect("model generation failed");
    Ok(PyLogLikelihoodModel {
        model: Arc::new(model),
        p_data_g_theta: None,
        p_theta_g_data: None,
    })
}

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
