//! Construction of Python Binding
mod mixture_models;
mod moves;
mod scientific;
mod toy;

use std::sync::Arc;

use numpy::{PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyTuple};

use crate::{
    ess::EnsembleSliceSampler,
    log_likelihood::{LogLikelihoodModel, ProbDataGivenThetaModel, ProbThetaGivenDataModel},
    moves::DynEnsembleMove,
    python_bindings::{
        mixture_models::{dyn_ftheta_ncomp_mm, dyn_mixture_model},
        moves::{diffconv_move, differential_move, hybrid_ddc_move},
        scientific::{normal_1d_deconv, normal_2d_gmm_deconv},
        toy::{neals_funnel, normal, normal_1d_mix, rosenbrock},
    },
};

#[pyclass]
pub struct PyLogLikelihoodModel {
    // Own the configured Rust model.
    model: Arc<dyn LogLikelihoodModel>,
    p_data_g_theta: Option<Arc<dyn ProbDataGivenThetaModel>>,
    p_theta_g_data: Option<Arc<dyn ProbThetaGivenDataModel>>,
}

#[pyclass]
pub struct PyDynMoveHandler {
    inner: Box<dyn DynEnsembleMove>,
}

#[pyclass(name = "EnsembleConfig")]
pub struct PyEnsembleConfig {
    n_samples: usize,
    n_walkers: usize,
    n_parameters: usize,
    n_burn_in: usize,
    n_threads: Option<usize>,
    n_thinning_per_sample: Option<usize>,
}

#[pymethods]
impl PyEnsembleConfig {
    #[new]
    #[pyo3(
        signature = (
            n_samples,
            n_walkers,
            n_parameters,
            n_burn_in,
            n_threads=None,
            n_thinning_per_sample=None
        ),
        text_signature = "(n_samples, n_walkers, n_parameters, n_burn_in, n_threads=None, n_thinning_per_sample=None)"
    )]
    fn new(
        n_samples: usize,
        n_walkers: usize,
        n_parameters: usize,
        n_burn_in: usize,
        n_threads: Option<usize>,
        n_thinning_per_sample: Option<usize>,
    ) -> PyResult<Self> {
        if n_samples == 0 {
            return Err(PyValueError::new_err(
                "max_n_samples must be greater than zero",
            ));
        }

        Ok(Self {
            n_samples,
            n_walkers,
            n_parameters,
            n_burn_in,
            n_threads,
            n_thinning_per_sample,
        })
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.n_samples
    }

    #[setter]
    fn set_n_samples(&mut self, value: usize) -> PyResult<()> {
        if value == 0 {
            return Err(PyValueError::new_err(
                "max_n_samples must be greater than zero",
            ));
        }
        self.n_samples = value;
        Ok(())
    }

    #[getter]
    fn n_walkers(&self) -> usize {
        self.n_walkers
    }

    #[setter]
    fn set_n_walkers(&mut self, value: usize) {
        self.n_walkers = value;
    }

    #[getter]
    fn n_parameters(&self) -> usize {
        self.n_parameters
    }

    #[setter]
    fn set_n_parameters(&mut self, value: usize) {
        self.n_parameters = value;
    }

    #[getter]
    fn n_burn_in(&self) -> usize {
        self.n_burn_in
    }

    #[setter]
    fn set_n_burn_in(&mut self, value: usize) {
        self.n_burn_in = value;
    }

    #[getter]
    fn n_threads(&self) -> Option<usize> {
        self.n_threads
    }

    #[setter]
    fn set_n_threads(&mut self, value: usize) {
        self.n_threads = Some(value);
    }

    #[getter]
    fn n_thinning_per_sample(&self) -> Option<usize> {
        self.n_thinning_per_sample
    }

    #[setter]
    fn set_n_thinning_per_sample(&mut self, value: usize) {
        self.n_thinning_per_sample = Some(value);
    }
}

#[pyfunction]
#[pyo3(
    signature = (
        log_likelihood,
        config,
        burn_in_move,
        sampling_move,
        initial_conditions
    ),
    text_signature = "(log_likelihood, config, burn_in_move, sampling_move, initial_conditions)"
)]
#[doc = "Run ensemble slice sampler MCMC."]
#[doc = ""]
#[doc = "Args:"]
#[doc = "    log_likelihood: The model being fit."]
#[doc = "    config: Sampler configuration specifying the number of steps, walkers, parameters, and burn-in steps."]
#[doc = "    burn_in_move: Move handler used during burn-in."]
#[doc = "    sampling_move: Move handler used for posterior sampling."]
#[doc = "    initial_conditions: Initial conditions from which the fit will be launched."]
#[doc = ""]
#[doc = "Returns:"]
#[doc = "    A tuple containing the chains (n_samples, N_WALKERS, N_PARAMETERS)"]
#[doc = "    and log-likelihood values (n_samples, N_WALKERS)."]
fn run_mcmc<'py>(
    py: Python<'py>,
    log_likelihood: PyRef<'_, PyLogLikelihoodModel>,
    config: PyRef<'_, PyEnsembleConfig>,
    burn_in_move: PyRef<'_, PyDynMoveHandler>,
    sampling_move: PyRef<'_, PyDynMoveHandler>,
    initial_conditions: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyTuple>> {
    let initial_conditions = initial_conditions.as_array().to_owned();

    let n_walkers = initial_conditions.nrows();
    let n_parameters = initial_conditions.ncols();

    if n_parameters != log_likelihood.model.get_n_parameters() {
        return Err(PyValueError::new_err(
            "Initial-condition columns must match model parameters",
        ));
    }

    if n_walkers < (2 * n_parameters).max(3) {
        return Err(PyValueError::new_err(
            "Too few walkers for the number of parameters",
        ));
    }

    // Copy the Rust selection values out of the borrowed Python
    // wrappers. The sampler owns these values for the whole run.
    let model = &log_likelihood.model;
    let burn = &burn_in_move.inner;
    let sampling = &sampling_move.inner;

    let mut sampler = EnsembleSliceSampler::new(
        config.n_samples,
        config.n_walkers,
        config.n_parameters,
        config.n_threads,
        config.n_thinning_per_sample,
    );

    sampler
        .initialise_dyn(model.as_ref(), initial_conditions.view())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    sampler
        .run_burn_in_dyn(model.as_ref(), burn.as_ref(), config.n_burn_in)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    sampler
        .run_sampler_dyn(model.as_ref(), sampling.as_ref())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // cast everything back to python for return
    let py_state = PyArray3::from_owned_array(py, sampler.extract_state_history());
    let py_ll = PyArray2::from_owned_array(py, sampler.extract_log_likelihood_history());

    let py_out = PyTuple::new(py, [py_state.into_any(), py_ll.into_any()])?;

    Ok(py_out)
}

#[pymodule]
fn _essrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core Python wrapper classes
    m.add_class::<PyLogLikelihoodModel>()?;
    // m.add_class::<PyProbDataGivenThetaModel>()?;
    m.add_class::<PyEnsembleConfig>()?;
    m.add_class::<PyDynMoveHandler>()?;

    // High-level sampler entry point
    m.add_function(wrap_pyfunction!(run_mcmc, m)?)?;

    // Model factories
    let model = PyModule::new(m.py(), "model")?;

    let toy = PyModule::new(m.py(), "toy")?;
    toy.add_function(wrap_pyfunction!(neals_funnel, &toy)?)?;
    toy.add_function(wrap_pyfunction!(normal, &toy)?)?;
    toy.add_function(wrap_pyfunction!(normal_1d_mix, &toy)?)?; // TODO: replace this with generic optimised implementation
    toy.add_function(wrap_pyfunction!(rosenbrock, &toy)?)?;

    toy.add_function(wrap_pyfunction!(dyn_mixture_model, &toy)?)?;

    let scientific = PyModule::new(m.py(), "scientific")?;
    scientific.add_function(wrap_pyfunction!(normal_1d_deconv, &scientific)?)?;
    scientific.add_function(wrap_pyfunction!(normal_2d_gmm_deconv, &scientific)?)?;

    scientific.add_function(wrap_pyfunction!(dyn_ftheta_ncomp_mm, &scientific)?)?;

    model.add_submodule(&toy)?;
    model.add_submodule(&scientific)?;

    m.add_submodule(&model)?;

    // Move factories
    let moves = PyModule::new(m.py(), "moves")?;
    moves.add_function(wrap_pyfunction!(differential_move, &moves)?)?;
    moves.add_function(wrap_pyfunction!(diffconv_move, &moves)?)?;
    moves.add_function(wrap_pyfunction!(hybrid_ddc_move, &moves)?)?;

    m.add_submodule(&moves)?;

    Ok(())
}
