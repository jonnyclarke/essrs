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
#[doc = "    `alpha`: Shape parameter shifting distribution towards 1."]
#[doc = "    `beta`: Shape parameter shifting distribution towards 0."]
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
#[
    pyo3(
        signature = (
            theta,
            list_models,
        ),
        text_signature = "theta, list_models"
    )
]
#[doc = "Dynamic mixture model sampler."]
#[doc = ""]
#[doc = "This allows you to bind together multiple sub models, e.g. normal 1d, into a mixture model."]
#[doc = ""]
#[doc = "Args:"]
#[doc = "    `theta`: Mixture weights. There must be n `theta` values where `n+1=m` with m being the number of model components (the final component has weight `1 - sum(theta)`). Should all be between `[0,1]` and should sum to <1"]
#[doc = "    `list_models`: List of models which share the same dimensionallity."]
pub fn dyn_ncomp_mix_model_sampler<'py>(
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
#[
    pyo3(
        signature = (),
        text_signature = "()"
    )
]
#[doc = "Neal's Funnel toy model."]
#[doc = ""]
#[doc = "Tests MCMC ability to sample hierarchical models with order of magnitude scale changes."]
pub fn neals_funnel() -> PyLogLikelihoodModel {
    let model = Arc::new(AnalyticNealsFunnelToyLogL::new());
    PyLogLikelihoodModel {
        model: model.clone(),
        sampler_component: Some(model),
    }
}

#[pyfunction]
#[
    pyo3(
        signature = (
            mu=0.0,
            sigma=1.0
        ),
        text_signature = "(mu: float = 0.0, sigma: float = 1.0)"
    )
]
#[doc = "Normal distribution -1 dimension- toy model."]
#[doc = ""]
#[doc = "Test of MCMC sampling routine base functionallity."]
#[doc = ""]
#[doc = "Args:"]
#[doc = "    `mu`: Centroid of normal distribution"]
#[doc = "    `sigma`: Standard deviation of normal distribution"]
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
#[
    pyo3(
        signature = (
            n_dimensions,
            mu,
            sigma,
            correlation
        ),
        text_signature = "(n_dimensions: int, mu: np.ndarray[float], sigma: np.ndarray[float), correlation: np.ndarray[float]"
    )
]
#[doc = "Normal distribution -N dimensional- toy model."]
#[doc = ""]
#[doc = "Test of MCMC sampling in high dimensional spaces."]
#[doc = ""]
#[doc = "Args:"]
#[doc = "    `n_dimensions`: The dimensionallity of the normal distribution. Must be >=1."]
#[doc = "    `mu`: Centroid vector of normal distribution. Must have length `n_dimensions`."]
#[doc = "    `sigma`: Standard deviation vector of normal distribution. Musthave length `n_dimensions`."]
#[doc = "    `correlation`: Dimensional correlations. Format: [p01, p02, p12, p03, p13, p23, p04, ...]. Must have length = `n_dimensions * (n_dimensions - 1) / 2`"]
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
#[
    pyo3(
        signature = (
            n,
            theta,
            mu,
            sigma,
        ),
        text_signature = "(n: int, theta: np.ndarray[float], mu: np.ndarray[float], sigma: np.ndarray[float)"
    )
]
#[doc = "Pre-configured mixture model of normal-1d distributions."]
#[doc = ""]
#[doc = "Test performance difference in dynamic model construction vs static construction."]
#[doc = ""]
#[doc = "Args:"]
#[doc = "    n: The number of normal distributions in the mixture."]
#[doc = "    theta: The mixture weights. Length = n - 1. Must sum to <1. All values should be between 0 and 1."]
#[doc = "    mu: Centroid vector of normal distributions. Must have length `n`."]
#[doc = "    sigma: Standard deviation vector of normal distributions. Musthave length `n`."]
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
