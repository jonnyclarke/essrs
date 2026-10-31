use std::sync::Arc;

use anyhow::Context;
use pyo3::prelude::*;

use crate::{
    log_likelihood::f_scientific::DynNcompMixtureModel,
    python_bindings::PyLogLikelihoodModel,
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
