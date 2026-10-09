use numpy::{PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{prelude::*, types::PyTuple};

use crate::{
    ess::EnsembleSliceSampler,
    log_likelihood::f_community::clarke2027::Clarke2027LogLiViracV2Density,
    moves::{
        diffconv_move::DiffConvMove, differential_move::DifferentialMove, hybrid::HybridOneInNMove,
    },
};

#[pyfunction]
pub fn virac2_rho_rc_rgbb_2peak<'py>(
    py: Python<'py>,
    max_n_steps: usize,
    n_burn_in: usize,
    initial_conditions: PyReadonlyArray2<'py, f64>,
    ks_data: PyReadonlyArray1<'py, f64>,
    ks_min: f64,
    ks_max: f64,
) -> PyResult<Bound<'py, PyTuple>> {
    let initial_conditions = initial_conditions.as_array().to_owned();
    let n_walkers = initial_conditions.nrows();
    let n_parameters: usize = initial_conditions.ncols();

    let d = ks_data.as_array().to_owned();

    let move_handler = DifferentialMove::default();
    let burn_move_handler = HybridOneInNMove::new(DifferentialMove::new(), DiffConvMove::new(), 10);
    let model = Clarke2027LogLiViracV2Density::new(d, ks_min, ks_max);

    let mut ess = EnsembleSliceSampler::new(max_n_steps, n_walkers, n_parameters);

    ess.initialise(&model, initial_conditions.view()).unwrap(); // compute log-likelihood of initial positions
    ess.run_burn_in(&model, &burn_move_handler, n_burn_in)
        .unwrap();
    ess.run_sampler(&model, &move_handler).unwrap(); // run sampler

    let chains = ess.extract_state_history();
    let ll_final = ess.extract_log_likelihood_history();

    let py_chains = PyArray3::from_owned_array(py, chains);
    let py_ll_final = PyArray2::from_owned_array(py, ll_final);
    let tup = PyTuple::new(py, [py_chains.into_any(), py_ll_final.into_any()])?;

    Ok(tup)
}
