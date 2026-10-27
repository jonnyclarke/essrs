//! Create python wrappers for the move sets

use pyo3::prelude::*;

use crate::{
    moves::{
        diffconv_move::DiffConvMove, differential_move::DifferentialMove,
        gaussian_move::GaussianMove, hybrid::DynHybridOneInNMove,
    },
    python_bindings::PyDynMoveHandler,
};

#[pyfunction]
pub fn differential_move() -> PyDynMoveHandler {
    PyDynMoveHandler {
        inner: Box::new(DifferentialMove::new()),
    }
}

#[pyfunction]
pub fn diffconv_move() -> PyDynMoveHandler {
    PyDynMoveHandler {
        inner: Box::new(DiffConvMove::new()),
    }
}

#[pyfunction]
pub fn gaussian_move() -> PyDynMoveHandler {
    PyDynMoveHandler {
        inner: Box::new(GaussianMove::new()),
    }
}

#[pyfunction]
pub fn hybrid_ddc_move(one_in_n: usize) -> PyDynMoveHandler {
    PyDynMoveHandler {
        inner: Box::new(DynHybridOneInNMove::new(
            Box::new(DifferentialMove::new()),
            Box::new(DiffConvMove::new()),
            one_in_n,
        )),
    }
}
