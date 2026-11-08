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
#[
    pyo3(
        signature = (
            values,
            errors,
        ),
        text_signature = "(value: numpy.ndarray[float], error: numpy.ndarray[float])"
    )
]
#[doc = "Normal distribution in 1 dimension with error deconvolution."]
#[doc = ""]
#[doc = "This enables the uses to extract the true underlying distribution of data which have been convolved with their errors."]
#[doc = ""]
#[doc = "Args:"]
#[doc = "    values: Data point values."]
#[doc = "    errors: Data point errors."]
pub fn normal_1d_deconv(
    values: PyReadonlyArray1<f64>,
    errors: PyReadonlyArray1<f64>,
) -> PyLogLikelihoodModel {
    let values = values.as_array().to_owned();
    let errors = errors.as_array().to_owned();

    let model = Arc::new(GaussianLl1dDataErrors::from_values_errors(&values, &errors));

    PyLogLikelihoodModel {
        model: model.clone(),
        sampler_component: None,
    }
}

#[pyfunction]
#[
    pyo3(
        signature = (
            values,
            errors,
            n,
        ),
        text_signature = "(value: numpy.ndarray[float], error: numpy.ndarray[float]), n: int"
    )
]
#[doc = "Normal distribution mixture model in 1 dimension with error deconvolution."]
#[doc = ""]
#[doc = "This enables the uses to extract the true underlying distribution of data which have been convolved with their errors."]
#[doc = ""]
#[doc = "Args:"]
#[doc = "    values: Data point values."]
#[doc = "    errors: Data point errors."]
#[doc = "    n: The number of normal distributions to include in the fit."]
pub fn normal_1d_mix_deconv(
    values: PyReadonlyArray1<f64>,
    errors: PyReadonlyArray1<f64>,
    n: usize,
) -> PyResult<PyLogLikelihoodModel> {
    let values = values.as_array().to_owned();
    let errors = errors.as_array().to_owned();

    let model = Arc::new(Normal1DimMixErrDeconv::new(values, errors, n)?);

    Ok(PyLogLikelihoodModel {
        model: model.clone(),
        sampler_component: None,
    })
}

#[pyfunction]
#[
    pyo3(
        signature = (
            n_components,
            data,
        ),
        text_signature = "(n_components: int, data: numpy.ndarray[float])"
    )
]
#[doc = "Normal distribution mixture model in 2 dimensions with error deconvolution."]
#[doc = ""]
#[doc = "This enables the uses to extract the true underlying distribution of data which have been convolved with their errors."]
#[doc = ""]
#[doc = "Args:"]
#[doc = "    `n_components`: The number of 2d normal distributions to include in the fit."]
#[doc = "    `data`: Data to fit. This should have shape `[n_data, 5]`. The data columns are `[x, y, xerr, yerr, p]`"]
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
