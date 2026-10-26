use numpy::{PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::moves::differential_move::DifferentialMove;
use crate::{
    chains::static_buffer::StaticBuffer, // uses to store the walker positions with each iteration
    ess::EnsembleSliceSampler,           // ensemble slice sampler struct
    log_likelihood::toy::neals_funnel::{
        AnalyticNealsFunnelToyLogL, // Likelihood model for 1d Gaussian data with errors
    },
};

#[pyfunction]
pub fn toy_neals_funnel<'py>(
    py: Python<'py>,
    max_n_steps: usize,
    n_burn_in: usize,
    initial_conditions: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let initial_conditions = initial_conditions.as_array().to_owned();
    let n_walkers = initial_conditions.nrows();
    let n_parameters: usize = initial_conditions.ncols();

    let chains = StaticBuffer::new(max_n_steps, n_walkers, n_parameters);
    let move_handler = DifferentialMove::new();
    let model = AnalyticNealsFunnelToyLogL::new();

    let mut ess =
        EnsembleSliceSampler::<StaticBuffer, DifferentialMove, AnalyticNealsFunnelToyLogL>::new(
            max_n_steps,
            n_walkers,
            n_parameters,
            chains,
            move_handler,
            model,
        );

    ess.initialise(initial_conditions.view()).unwrap(); // compute log-likelihood of initial positions
    ess.run_burn_in(n_burn_in, &DifferentialMove::new())
        .unwrap();
    ess.run_sampler().unwrap(); // run sampler

    let result = ess.extract_state();

    Ok(PyArray3::from_owned_array(py, result))
}
