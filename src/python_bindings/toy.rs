//! Python log-likelihood factory functions; toy models

use std::sync::Arc;

use anyhow::Context;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::{
    log_likelihood::f_toy::{
        AnalyticBetaDistToyLogL, AnalyticNealsFunnelToyLogL, AnalyticRosenbrockToyLogL,
        DynToyMixModelDistributionSampler,
        normal::{
            AnalyticNormalToyLogL, BuildAnalyticNormalMixtureOneD, BuildAnalyticNormalNDimensional,
        },
    },
    python_bindings::PyLogLikelihoodModel,
};

#[pyfunction]
#[
    pyo3(
        signature = (
            alpha=0.5,
            beta=0.5
        ),
        text_signature = "(alpha=0.5, beta=0.5)"
    )
]
#[doc = "Beta Distribution toy model."]
#[doc = ""]
#[doc = "Tests MCMC ability to recover distribution near an asymptotic boundary."]
#[doc = ""]
#[doc = "Args:"]
#[doc = "    alpha: Shape parameter shifting distribution towards 1."]
#[doc = "    beta: Shape parameter shifting distribution towards 0."]
pub fn beta_distribution(alpha: f64, beta: f64) -> PyResult<PyLogLikelihoodModel> {
    let model = Arc::new(
        AnalyticBetaDistToyLogL::new(alpha, beta)
            .context("construction of 'beta_distribution' python binding has failed")?,
    );
    Ok(PyLogLikelihoodModel {
        model: model.clone(),
        sampler_component: Some(model),
    })
}

#[pyfunction]
pub fn dyn_fdata_ncomp_mm<'py>(
    theta: PyReadonlyArray1<'py, f64>,
    list_models: Vec<PyRef<'_, PyLogLikelihoodModel>>,
) -> PyResult<PyLogLikelihoodModel> {
    let theta = theta.as_array().to_owned();
    let components = list_models
        .iter()
        .map(|model| {
            model
                .sampler_component
                .as_ref()
                .cloned()
                .context("Model does not have a distribution sampler")
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
        })
        .collect::<PyResult<Vec<_>>>()?;

    Ok(PyLogLikelihoodModel {
        model: Arc::new(
            DynToyMixModelDistributionSampler::new(theta, components)
                .context("failed to build dynamic mixture model")?,
        ),
        sampler_component: None,
    })
}

#[pyfunction]
pub fn neals_funnel() -> PyLogLikelihoodModel {
    let model = Arc::new(AnalyticNealsFunnelToyLogL::new());
    PyLogLikelihoodModel {
        model: model.clone(),
        sampler_component: Some(model),
    }
}

#[pyfunction]
pub fn normal_1d(mu: f64, sigma: f64) -> PyResult<PyLogLikelihoodModel> {
    let model = Arc::new(
        AnalyticNormalToyLogL::new(mu, sigma)
            .context("construction of 'normal_1d' python binding has failed")?,
    );
    Ok(PyLogLikelihoodModel {
        model: model.clone(),
        sampler_component: Some(model),
    })
}

#[pyfunction]
pub fn normal_nd(
    n_dimensions: usize,
    mu: PyReadonlyArray1<f64>,
    sigma: PyReadonlyArray1<f64>,
    correlation: PyReadonlyArray1<f64>,
) -> PyResult<PyLogLikelihoodModel> {
    Ok(PyLogLikelihoodModel {
        model: Arc::new(
            BuildAnalyticNormalNDimensional::new(n_dimensions)
                .mu(mu.as_array().to_owned())
                .sigma(sigma.as_array().to_owned())
                .correlation(correlation.as_array().to_owned())
                .build()
                .context("normal n-dimensional model build has failed")?,
        ),
        sampler_component: None,
    })
}

#[pyfunction]
pub fn normal_1d_mix(
    n: usize,
    theta: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    sigma: PyReadonlyArray1<f64>,
) -> PyResult<PyLogLikelihoodModel> {
    Ok(PyLogLikelihoodModel {
        model: Arc::new(
            BuildAnalyticNormalMixtureOneD::new(n)
                .theta(theta.as_array().to_owned())
                .mu(mu.as_array().to_owned())
                .sigma(sigma.as_array().to_owned())
                .build()
                .context("model build has failed")?,
        ),
        sampler_component: None,
    })
}

#[pyfunction]
#[
    pyo3(
        signature = (
            a=1.0,
            b=3.0
        ),
        text_signature = "(a=1.0, b=3.0)"
    )
]
#[doc = "Rosenbrock toy model."]
#[doc = ""]
#[doc = "Used for testing MCMC implementations due to challenging banana shaped posterior."]
#[doc = ""]
#[doc = "Args:"]
#[doc = "    a: Parameter controlling shifts in xy space."]
#[doc = "    b: Parameter controlling strength of banana curvature."]
pub fn rosenbrock(a: f64, b: f64) -> PyLogLikelihoodModel {
    let model = Arc::new(AnalyticRosenbrockToyLogL::new(a, b));
    PyLogLikelihoodModel {
        model: model.clone(),
        sampler_component: Some(model),
    }
}
