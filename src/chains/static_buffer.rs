mod final_state;
mod store_final_positions;

use ndarray::{Array1, Array2, Array3, Axis, s};

use crate::chains::ChainBuffer;

pub struct StaticBuffer {
    chains: Array3<f64>,
    n_stored: usize,
}

impl StaticBuffer {
    pub fn new(max_n_steps: usize, n_walkers: usize, n_parameters: usize) -> Self {
        Self {
            chains: Array3::<f64>::zeros((max_n_steps, n_walkers, n_parameters)),
            n_stored: 0,
        }
    }
}

impl ChainBuffer for StaticBuffer {
    fn record_state(&mut self, state: &Array2<f64>) {
        let mut slice = self.chains.index_axis_mut(Axis(0), self.n_stored);
        slice.assign(state);
        self.n_stored += 1;
    }

    fn extract_state(&self) -> Array3<f64> {
        self.chains.clone()
    }

    fn extract_parameter_history(&self, i_walker: usize, i_parameter: usize) -> Array1<f64> {
        self.chains.slice(s![.., i_walker, i_parameter]).to_owned()
    }

    fn display_parameter_summaries(&self, n_params: i32) {
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
    use crate::testing::make_test_static_buffer;

    // ---- Mock Config ----
    // type TestConfig = EnsembleSliceSamplerConfig<10, 2, 2>;

    #[test]
    fn test_record_and_extract_history() {
        let mut buffer = StaticBuffer::new(10, 2, 2);

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
        let buffer = StaticBuffer::new(10, 2, 2);

        let history = buffer.extract_parameter_history(0, 0);

        // No states recorded yet, so should be all zeros
        for value in history.iter() {
            assert_eq!(*value, 0.0);
        }
    }

    #[test]
    fn test_extract_state() {
        let buffer = make_test_static_buffer();

        assert_eq!(
            buffer.extract_state(),
            array![
                [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],
                [[21.0, 22.0], [23.0, 24.0], [25.0, 26.0], [27.0, 28.0]]
            ]
        )
    }

    #[test]
    fn test_display_parameter_summaries() {
        let buffer = make_test_static_buffer();
        buffer.display_parameter_summaries(2);
    }
}
