pub mod static_buffer;

use ndarray::{Array1, Array2};

use crate::FloatExt;

pub trait ChainBuffer<T: FloatExt> {
    fn record_state(&mut self, state: &Array2<T>);
    fn extract_parameter_history(&self, i_walker: usize, i_parameter: usize) -> Array1<T>;
}
