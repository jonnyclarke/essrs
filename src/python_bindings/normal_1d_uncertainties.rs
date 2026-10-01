use ndarray::{Axis, array, stack};
use numpy::{PyArray3, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::chains::ChainBuffer;
use crate::{
    chains::static_buffer::StaticBuffer, // uses to store the walker positions with each iteration
    ess::{EnsembleSliceSampler, EnsembleSliceSamplerConfig}, // ensemble slice sampler struct
    log_likelihood::gaussian_1d_data_err::{
        GaussianLl1dDataErrors, // Likelihood model for 1d Gaussian data with errors
    },
    moves::MoveHandler, // struct defining which moves to use
};

#[pyfunction]
fn uncertainty_deconvolution<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    error: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    // convert to internal...
    let data = data.as_array();
    let error = error.as_array();

    const N_STEPS: usize = 500; // number of steps for the walkers
    const N_PARAMETERS: usize = 2; // number of parameters in the fit: mean & standard deviation
    const N_WALKERS: usize = 12; // number of walkers used simultaneously NOTE: > 2 x N_PARAMETERS

    // Build config object which ensures all components are consistent
    type Config = EnsembleSliceSamplerConfig<N_STEPS, N_WALKERS, N_PARAMETERS>;

    let observations = stack![Axis(1), data, error];

    let chains = StaticBuffer::<Config>::new();
    let model = GaussianLl1dDataErrors::new(observations);
    let mut ess = EnsembleSliceSampler::<StaticBuffer<Config>, GaussianLl1dDataErrors, Config>::new(
        chains, model,
    );

    let start = array![
        [0.102, 0.87],
        [0.103, 0.76],
        [0.101, 0.90],
        [0.999, 0.65],
        [0.994, 0.81],
        [0.108, 1.01],
        [0.997, 0.72],
        [0.995, 0.79],
        [0.993, 0.67],
        [0.102, 0.51],
        [0.998, 1.12],
        [0.101, 0.31],
    ];

    ess.initialise(&start).unwrap(); // compute log-likelihood of initial positions
    let move_handler = MoveHandler::default(); // use default move setup 90% differential + 10% gaussian 
    ess.run_sampler(100, move_handler).unwrap(); // run sampler

    let result = ess.chains.extract_state();

    Ok(PyArray3::from_owned_array(py, result))
}

#[pymodule]
fn _essrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(uncertainty_deconvolution, m)?)?;
    Ok(())
}
