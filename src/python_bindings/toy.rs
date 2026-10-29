//! Python log-likelihood factory functions; toy models

use pyo3::prelude::*;

use crate::{
    log_likelihood::f_toy::{
        AnalyticNealsFunnelToyLogL, AnalyticNormalToyLogL, AnalyticRosenbrockToyLogL,
    },
    python_bindings::PyLogLikelihoodModel,
};

#[pyfunction]
pub fn neals_funnel() -> PyLogLikelihoodModel {
    PyLogLikelihoodModel {
        inner: Box::new(AnalyticNealsFunnelToyLogL::new()),
    }
}

#[pyfunction]
pub fn normal(mu: f64, sigma: f64) -> PyLogLikelihoodModel {
    PyLogLikelihoodModel {
        inner: Box::new(AnalyticNormalToyLogL::new(mu, sigma)),
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
    PyLogLikelihoodModel {
        inner: Box::new(AnalyticRosenbrockToyLogL::new(a, b)),
    }
}
