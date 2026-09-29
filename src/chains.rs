//! The chains module is where we track the history of the MCMC walkers.

pub mod static_buffer;

use ndarray::{Array1, Array2};

/// Trait containing all required methods of the chain storage struct.
pub trait ChainBuffer {
    /// This takes a 2D array [N- * N-] and places it into the chain storage.
    fn record_state(&mut self, state: &Array2<f64>);
    /// This extracts, for a given walker and a given parameter, the chain of values.
    fn extract_parameter_history(&self, i_walker: usize, i_parameter: usize) -> Array1<f64>;
    /// This function prints a summary of the parameters to the screen
    fn display_parameter_summaries(&self);
}
