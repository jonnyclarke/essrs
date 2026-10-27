//! Storage of MCMC chains and corresponding likelihood values.

/// Static buffer chain storage; 3D ndarray
pub mod static_buffer;

use ndarray::{Array2, Array3, ArrayViewMut1, ArrayViewMut2};
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

    fn increment_n_stored(&mut self);

    fn get_next_log_likelihood_vec<'a>(&'a mut self) -> ArrayViewMut1<'a, f64>;

    fn get_next_state_matrix<'a>(&'a mut self) -> ArrayViewMut2<'a, f64>;

    /// Extract the three-dimensional state matrix and output (can have empty buffer!)
    fn extract_state_history(&self) -> Array3<f64>;

    /// Extract the two dimensional log-likelihood history
    fn extract_log_likeihood_history(&self) -> Array2<f64>;

    /// This function prints a summary of the parameters to the screen
    fn display_parameter_summaries(&self, n_params: i32);
}
