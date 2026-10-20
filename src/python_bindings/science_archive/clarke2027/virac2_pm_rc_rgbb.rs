use numpy::{PyArray1, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{prelude::*, types::PyTuple};

use crate::{chains::ChainBuffer, moves::move_handlers::diffnorm_handler::DiffNormHandler};
use crate::{
    chains::static_buffer::StaticBuffer, // uses to store the walker positions with each iteration
    ess::EnsembleSliceSampler,           // ensemble slice sampler struct
    log_likelihood::science_archive::clarke2027::virac2_pm_rc_rgbb::{
        C2027LlViracV2ProperMotionRcRgbb, // Likelihood model for 1d Gaussian data with errors
    },
};

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn virac2_pm_rc_rgbb<'py>(
    py: Python<'py>,
    max_n_steps: usize,
    n_burn_in: usize,
    initial_conditions: PyReadonlyArray2<'py, f64>,
    b_coeffs: PyReadonlyArray2<'py, f64>,
    pm: PyReadonlyArray1<'py, f64>,
    pme: PyReadonlyArray1<'py, f64>,
    theta_rc: PyReadonlyArray1<'py, f64>,
    weight: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyTuple>> {
    let initial_conditions = initial_conditions.as_array().to_owned();
    let n_walkers = initial_conditions.nrows();
    let n_parameters: usize = initial_conditions.ncols();

    let b_coeffs = b_coeffs.as_array().to_owned();
    let pm = pm.as_array().to_owned();
    let pme = pme.as_array().to_owned();
    let theta_rc = theta_rc.as_array().to_owned();
    let weight = weight.as_array().to_owned();

    let chains = StaticBuffer::new(max_n_steps, n_walkers, n_parameters);
    let move_handler = DiffNormHandler::default();
    let model = C2027LlViracV2ProperMotionRcRgbb::new(b_coeffs, pm, pme, theta_rc, weight);

    let mut ess = EnsembleSliceSampler::<
        StaticBuffer,
        DiffNormHandler,
        C2027LlViracV2ProperMotionRcRgbb,
    >::new(
        max_n_steps,
        n_walkers,
        n_parameters,
        chains,
        move_handler,
        model,
    );

    ess.initialise(initial_conditions.view()).unwrap(); // compute log-likelihood of initial positions
    ess.run_sampler(n_burn_in).unwrap(); // run sampler

    let chains = ess.chains.extract_state();
    let ll_final = ess.final_log_likelihood().to_owned();

    let py_chains = PyArray3::from_owned_array(py, chains);
    let py_ll_final = PyArray1::from_owned_array(py, ll_final);

    let tup = PyTuple::new(py, [py_chains.into_any(), py_ll_final.into_any()])?;

    Ok(tup)
}
