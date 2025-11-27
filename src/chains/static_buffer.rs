use std::marker::PhantomData;

use ndarray::{Array1, Array2, Array3, Axis, s};

use crate::{FloatExt, chains::ChainBuffer, ess::EnsembleSliceSamplerConfigTrait};

pub struct StaticBuffer<T, CONFIG: EnsembleSliceSamplerConfigTrait> {
    chains: Array3<T>,
    n_stored: usize,
    _config: PhantomData<CONFIG>,
}

impl<T: FloatExt, CONFIG: EnsembleSliceSamplerConfigTrait> StaticBuffer<T, CONFIG> {
    pub fn new() -> Self {
        Self {
            chains: Array3::<T>::zeros((
                CONFIG::MAX_N_STEPS,
                CONFIG::N_WALKERS,
                CONFIG::N_PARAMETERS,
            )),
            n_stored: 0,
            _config: PhantomData,
        }
    }
}

impl<T: FloatExt, CONFIG: EnsembleSliceSamplerConfigTrait> Default for StaticBuffer<T, CONFIG> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatExt, CONFIG: EnsembleSliceSamplerConfigTrait> ChainBuffer<T>
    for StaticBuffer<T, CONFIG>
{
    fn record_state(&mut self, state: &Array2<T>) {
        let mut slice = self.chains.index_axis_mut(Axis(0), self.n_stored);
        slice.assign(state);
        self.n_stored += 1;
    }

    fn extract_parameter_history(&self, i_walker: usize, i_parameter: usize) -> Array1<T> {
        self.chains.slice(s![.., i_walker, i_parameter]).to_owned()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, array};

    use super::*;
    use crate::ess::EnsembleSliceSamplerConfig;

    // ---- Mock Config ----
    type TestConfig = EnsembleSliceSamplerConfig<10, 2, 2, 3, 5, 1, false>;

    #[test]
    fn test_record_and_extract_history() {
        let mut buffer = StaticBuffer::<f64, TestConfig>::default();

        // Step 0
        let state0: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0],];
        buffer.record_state(&state0);

        // Step 1
        let state1: Array2<f64> = array![[5.0, 6.0], [7.0, 8.0],];
        buffer.record_state(&state1);

        // Extract history for walker 0, parameter 0
        let history = buffer.extract_parameter_history(0, 0);

        assert_eq!(history[0], 1.0);
        assert_eq!(history[1], 5.0);

        // Extract history for walker 1, parameter 1
        let history = buffer.extract_parameter_history(1, 1);

        assert_eq!(history[0], 4.0);
        assert_eq!(history[1], 8.0);
    }

    #[test]
    fn test_initial_state_is_zero() {
        let buffer = StaticBuffer::<f64, TestConfig>::new();

        let history = buffer.extract_parameter_history(0, 0);

        // No states recorded yet, so should be all zeros
        for value in history.iter() {
            assert_eq!(*value, 0.0);
        }
    }
}
