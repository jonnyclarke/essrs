//! Chains are the memory of previous values in the MCMC

/// Static buffer chain storage; 3D ndarray
pub mod static_buffer;

use ndarray::{Array1, Array2, Array3};
use thiserror::Error;

/// Errors accessing the chain
#[derive(Debug, Error)]
pub enum ChainError {
    /// User attempts to access an empty chain buffer
    #[error("invalid access to empty chain buffer")]
    EmptyChainBufferError,
}

/// Trait containing all required methods of the chain storage struct.
pub trait ChainBuffer {
    /// Return the max number of steps
    fn get_n_steps(&self) -> usize;

    /// Return the number of walkers
    fn get_n_walkers(&self) -> usize;

    /// Return the number of parameters;
    fn get_n_parameters(&self) -> usize;

    /// This takes a 2D array [N- * N-] and places it into the chain storage.
    fn record_state(&mut self, state: &Array2<f64>);

    /// Extract the three-dimensional state matrix and output (can have empty buffer!)
    fn extract_state(&self) -> Array3<f64>;

    /// This extracts, for a given walker and a given parameter, the chain of values.
    fn extract_parameter_history(&self, i_walker: usize, i_parameter: usize) -> Array1<f64>;

    /// This function prints a summary of the parameters to the screen
    fn display_parameter_summaries(&self, n_params: i32);
}
