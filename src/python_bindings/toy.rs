//! Python log-likelihood factory functions; toy models

use std::sync::Arc;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::{
    log_likelihood::f_toy::{
        AnalyticNealsFunnelToyLogL, AnalyticNormalToyLogL, AnalyticRosenbrockToyLogL,
        BuildAnalyticNormalMixtureOneD,
    },
    python_bindings::PyLogLikelihoodModel, //, PyProbDataGivenThetaModel},
};

#[pyfunction]
pub fn neals_funnel() -> PyLogLikelihoodModel {
    let model = Arc::new(AnalyticNealsFunnelToyLogL::new());
    PyLogLikelihoodModel {
        model: model.clone(),
        p_data_g_theta: Some(model),
        p_theta_g_data: None,
    }
}

#[pyfunction]
pub fn normal(mu: f64, sigma: f64) -> PyLogLikelihoodModel {
    let model = Arc::new(AnalyticNormalToyLogL::new(mu, sigma));
    PyLogLikelihoodModel {
        model: model.clone(),
        p_data_g_theta: Some(model),
        p_theta_g_data: None,
    }
}

#[pyfunction]
pub fn normal_1d_mix(
    n: usize,
    theta: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    sigma: PyReadonlyArray1<f64>,
) -> PyLogLikelihoodModel {
    PyLogLikelihoodModel {
        model: Arc::new(
            BuildAnalyticNormalMixtureOneD::new(n)
                .theta(theta.as_array().to_owned())
                .mu(mu.as_array().to_owned())
                .sigma(sigma.as_array().to_owned())
                .build(),
        ),
        p_data_g_theta: None,
        p_theta_g_data: None,
    }
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
        p_data_g_theta: Some(model),
        p_theta_g_data: None,
    }
}
