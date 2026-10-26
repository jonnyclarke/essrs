//! Construction of Python Binding
mod moves;
mod scientific;
mod toy;

use numpy::{PyArray3, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::{
    ess::EnsembleSliceSampler,
    log_likelihood::LogLikelihoodModel,
    moves::DynEnsembleMove,
    python_bindings::{
        moves::{diffconv_move, differential_move, gaussian_move, hybrid_ddc_move},
        scientific::{normal_1d_deconv, normal_2d_gmm_deconv},
        toy::{neals_funnel, normal, rosenbrock},
    },
};

#[pyclass]
pub struct PyLogLikelihoodModel {
    // Own the configured Rust model.
    inner: Box<dyn LogLikelihoodModel>,
}

#[pyclass]
pub struct PyDynMoveHandler {
    inner: Box<dyn DynEnsembleMove>,
}

#[pyclass(name = "EnsembleConfig")]
pub struct PyEnsembleConfig {
    n_steps: usize,
    n_walkers: usize,
    n_parameters: usize,
    n_burn_in: usize,
}

#[pymethods]
impl PyEnsembleConfig {
    #[new]
    fn new(
        n_steps: usize,
        n_walkers: usize,
        n_parameters: usize,
        n_burn_in: usize,
    ) -> PyResult<Self> {
        if n_steps == 0 {
            return Err(PyValueError::new_err(
                "max_n_steps must be greater than zero",
            ));
        }

        Ok(Self {
            n_steps,
            n_walkers,
            n_parameters,
            n_burn_in,
        })
    }

    #[getter]
    fn n_steps(&self) -> usize {
        self.n_steps
    }

    #[setter]
    fn set_n_steps(&mut self, value: usize) -> PyResult<()> {
        if value == 0 {
            return Err(PyValueError::new_err(
                "max_n_steps must be greater than zero",
            ));
        }
        self.n_steps = value;
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
}

#[pyfunction]
fn run_mcmc<'py>(
    py: Python<'py>,
    log_likelihood: PyRef<'_, PyLogLikelihoodModel>,
    config: PyRef<'_, PyEnsembleConfig>,
    burn_in_move: PyRef<'_, PyDynMoveHandler>,
    sampling_move: PyRef<'_, PyDynMoveHandler>,
    initial_conditions: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let initial_conditions = initial_conditions.as_array().to_owned();

    let n_walkers = initial_conditions.nrows();
    let n_parameters = initial_conditions.ncols();

    if n_parameters != log_likelihood.inner.get_n_parameters() {
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
    let model = &log_likelihood.inner;
    let burn = &burn_in_move.inner;
    let sampling = &sampling_move.inner;

    let mut sampler =
        EnsembleSliceSampler::new(config.n_steps, config.n_walkers, config.n_parameters);

    sampler
        .initialise_dyn(model.as_ref(), initial_conditions.view())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    sampler
        .run_burn_in_dyn(model.as_ref(), burn.as_ref(), config.n_burn_in)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    sampler
        .run_sampler_dyn(model.as_ref(), sampling.as_ref())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyArray3::from_owned_array(py, sampler.extract_state()))
}

#[pymodule]
fn _essrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core Python wrapper classes
    m.add_class::<PyLogLikelihoodModel>()?;
    m.add_class::<PyEnsembleConfig>()?;
    m.add_class::<PyDynMoveHandler>()?;

    // High-level sampler entry point
    m.add_function(wrap_pyfunction!(run_mcmc, m)?)?;

    // Model factories
    let model = PyModule::new(m.py(), "model")?;

    let toy = PyModule::new(m.py(), "toy")?;
    toy.add_function(wrap_pyfunction!(neals_funnel, &toy)?)?;
    toy.add_function(wrap_pyfunction!(normal, &toy)?)?;
    toy.add_function(wrap_pyfunction!(rosenbrock, m)?)?;

    let scientific = PyModule::new(m.py(), "scientific")?;
    scientific.add_function(wrap_pyfunction!(normal_1d_deconv, &toy)?)?;
    scientific.add_function(wrap_pyfunction!(normal_2d_gmm_deconv, &toy)?)?;

    model.add_submodule(&toy)?;
    model.add_submodule(&scientific)?;

    m.add_submodule(&model)?;

    // Move factories
    let moves = PyModule::new(m.py(), "moves")?;
    moves.add_function(wrap_pyfunction!(differential_move, &moves)?)?;
    moves.add_function(wrap_pyfunction!(diffconv_move, &moves)?)?;
    moves.add_function(wrap_pyfunction!(gaussian_move, &moves)?)?;
    moves.add_function(wrap_pyfunction!(hybrid_ddc_move, &moves)?)?;

    m.add_submodule(&moves)?;

    Ok(())
}
