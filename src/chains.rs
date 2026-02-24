pub mod static_buffer;

use ndarray::{Array1, Array2};

pub trait ChainBuffer {
    fn record_state(&mut self, state: &Array2<f64>);
    fn extract_parameter_history(&self, i_walker: usize, i_parameter: usize) -> Array1<f64>;
    fn display_parameter_summaries(&self);
}
