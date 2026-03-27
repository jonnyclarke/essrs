mod final_state;
mod store_final_positions;

use std::marker::PhantomData;

use ndarray::{Array1, Array2, Array3, Axis, s};

use crate::{chains::ChainBuffer, ess::EnsembleSliceSamplerConfigTrait};

pub struct StaticBuffer<CONFIG: EnsembleSliceSamplerConfigTrait> {
    chains: Array3<f64>,
    n_stored: usize,
    _config: PhantomData<CONFIG>,
}

impl<CONFIG: EnsembleSliceSamplerConfigTrait> StaticBuffer<CONFIG> {
    pub fn new() -> Self {
        Self {
            chains: Array3::<f64>::zeros((
                CONFIG::MAX_N_STEPS,
                CONFIG::N_WALKERS,
                CONFIG::N_PARAMETERS,
            )),
            n_stored: 0,
            _config: PhantomData,
        }
    }

    // normal API uses private constructor/fields
    #[cfg(test)]
    pub fn test_new(chains: Array3<f64>, n_stored: usize) -> Self {
        assert_eq!(
            chains.shape(),
            &[CONFIG::MAX_N_STEPS, CONFIG::N_WALKERS, CONFIG::N_PARAMETERS]
        );
        assert_eq!(n_stored, CONFIG::MAX_N_STEPS);
        Self {
            chains,
            n_stored,
            _config: std::marker::PhantomData,
        }
    }
}

impl<CONFIG: EnsembleSliceSamplerConfigTrait> Default for StaticBuffer<CONFIG> {
    fn default() -> Self {
        Self::new()
    }
}

impl<CONFIG: EnsembleSliceSamplerConfigTrait> ChainBuffer for StaticBuffer<CONFIG> {
    fn record_state(&mut self, state: &Array2<f64>) {
        let mut slice = self.chains.index_axis_mut(Axis(0), self.n_stored);
        slice.assign(state);
        self.n_stored += 1;
    }

    fn extract_parameter_history(&self, i_walker: usize, i_parameter: usize) -> Array1<f64> {
        self.chains.slice(s![.., i_walker, i_parameter]).to_owned()
    }

    fn display_parameter_summaries(&self) {
        let n_params = CONFIG::N_PARAMETERS;

        println!(
            "{:<12} {:>12} {:>12} {:>12} {:>12}",
            "Parameter", "Mean", "StdDev", "Min", "Max"
        );

        for i_param in 0..n_params {
            // Get all values for this parameter: (n_stored, n_walkers)
            let values = self.chains.slice(s![..self.n_stored, .., i_param]);

            // Flatten into 1D array for statistics
            let flat_values = values.iter().copied().collect::<Vec<f64>>();

            if flat_values.is_empty() {
                continue;
            }

            let sum = flat_values.iter().copied().sum::<f64>();
            let mean = sum / (flat_values.len() as f64);

            // Standard deviation
            let var = flat_values
                .iter()
                .map(|v| (*v - mean) * (*v - mean))
                .sum::<f64>()
                / (flat_values.len() as f64);
            let std = var.sqrt();

            let min = flat_values.iter().copied().fold(flat_values[0], f64::min);
            let max = flat_values.iter().copied().fold(flat_values[0], f64::max);

            println!(
                "{:<12} {:>12.5} {:>12.5} {:>12.5} {:>12.5}",
                i_param, mean, std, min, max
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, array};

    use super::*;
    use crate::ess::EnsembleSliceSamplerConfig;

    // ---- Mock Config ----
    type TestConfig = EnsembleSliceSamplerConfig<10, 2, 2>;

    #[test]
    fn test_record_and_extract_history() {
        let mut buffer = StaticBuffer::<TestConfig>::default();

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
        let buffer = StaticBuffer::<TestConfig>::new();

        let history = buffer.extract_parameter_history(0, 0);

        // No states recorded yet, so should be all zeros
        for value in history.iter() {
            assert_eq!(*value, 0.0);
        }
    }
}
