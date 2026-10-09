use numpy::{PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::{chains::ChainBuffer, moves::move_handlers::differential_handler::DifferentialHandler};
use crate::{
    chains::static_buffer::StaticBuffer, // uses to store the walker positions with each iteration
    ess::EnsembleSliceSampler,           // ensemble slice sampler struct
    log_likelihood::science_archive::clarke2027::virac2_rho::{
        Clarke2027LogLiViracV2Density, // Likelihood model for 1d Gaussian data with errors
    },
};

#[pyfunction]
pub fn virac2_rho<'py>(
    py: Python<'py>,
    max_n_steps: usize,
    n_burn_in: usize,
    initial_conditions: PyReadonlyArray2<'py, f64>,
    ks_data: PyReadonlyArray1<'py, f64>,
    ks_min: f64,
    ks_max: f64
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let initial_conditions = initial_conditions.as_array().to_owned();
    let n_walkers = initial_conditions.nrows();
    let n_parameters: usize = initial_conditions.ncols();

    let d = ks_data.as_array().to_owned();

    let chains = StaticBuffer::new(max_n_steps, n_walkers, n_parameters);
    let move_handler = DifferentialHandler::new();
    let model = Clarke2027LogLiViracV2Density::new(d, ks_min, ks_max);

    let mut ess =
        EnsembleSliceSampler::<StaticBuffer, DifferentialHandler, Clarke2027LogLiViracV2Density>::new(
            max_n_steps,
            n_walkers,
            n_parameters,
            chains,
            move_handler,
            model,
        );

    ess.initialise(initial_conditions.view()).unwrap(); // compute log-likelihood of initial positions
    ess.run_sampler(n_burn_in).unwrap(); // run sampler

    let result = ess.chains.extract_state();

    Ok(PyArray3::from_owned_array(py, result))
}
