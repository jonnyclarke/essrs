//! Create python wrappers for the move sets

use pyo3::prelude::*;

use crate::{
    moves::{
        diffconv_move::DiffConvMove, differential_move::DifferentialMove,
        hybrid::DynHybridOneInNMove,
    },
    python_bindings::PyDynMoveHandler,
};

#[pyfunction]
#[doc = "Differential move handler"]
pub fn differential_move() -> PyDynMoveHandler {
    PyDynMoveHandler {
        inner: Box::new(DifferentialMove::new()),
    }
}

#[pyfunction]
#[doc = "Differential - Convergence move handler"]
#[doc = ""]
#[doc = "WARNING: this should not be used for sampling posterior probability"]
pub fn diffconv_move() -> PyDynMoveHandler {
    PyDynMoveHandler {
        inner: Box::new(DiffConvMove::new()),
    }
}

#[pyfunction]
#[
    pyo3(
        signature = (
            one_in_n=10,
        ),
        text_signature = "(one_in_n=10)"
    )
]
#[doc = "Hybrid diff - (diff-conv) move"]
#[doc = ""]
#[doc = "This move runs a single `differential convergence` move for every `n-1` differential moves."]
#[doc = "This can be beneficial in that, during burn in phase, it can help walkers stuck in low probability regions to join with main group"]
#[doc = ""]
#[doc = "Args:"]
#[doc = "   `one_in_n`: The number of moves comprising one `set`. A set consists of `n-1` differential move followed by a single `diff-conv` move."]
pub fn hybrid_ddc_move(one_in_n: usize) -> PyDynMoveHandler {
    PyDynMoveHandler {
        inner: Box::new(DynHybridOneInNMove::new(
            Box::new(DifferentialMove::new()),
            Box::new(DiffConvMove::new()),
            one_in_n,
        )),
    }
}
