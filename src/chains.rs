//! The chains module is where we track the history of the MCMC walkers.

pub mod static_buffer;

use ndarray::{Array1, Array2, Array3};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ChainError {
    #[error("invalid access to empty chain buffer")]
    EmptyChainBufferError,
}

/// Trait containing all required methods of the chain storage struct.
pub trait ChainBuffer {
    /// This takes a 2D array [N- * N-] and places it into the chain storage.
    fn record_state(&mut self, state: &Array2<f64>);
    // Extract the three-dimensional state matrix and output (can have empty buffer!)
    fn extract_state(&self) -> Array3<f64>;
    /// This extracts, for a given walker and a given parameter, the chain of values.
    fn extract_parameter_history(&self, i_walker: usize, i_parameter: usize) -> Array1<f64>;
    /// This function prints a summary of the parameters to the screen
    fn display_parameter_summaries(&self);
}
