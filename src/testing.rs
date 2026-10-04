use anyhow;
use ndarray::{Array1, Array2, array};

use crate::{
    chains::{ChainBuffer, static_buffer::StaticBuffer},
    log_likelihood::LogLikelihoodModel,
    state::WalkerState,
};

pub struct WalkerStateBuilder {
    state: WalkerState,
}

impl WalkerStateBuilder {
    pub fn new() -> Self {
        Self {
            state: WalkerState::new(3, 1),
        }
    }

    pub fn set_state_matrix(mut self, x: &Array2<f64>) -> Self {
        self.state.get_mut_state_matrix().assign(x);

        self
    }

    pub fn set_ll_array(mut self, ll: &Array1<f64>) -> Self {
        self.state.get_mut_ll_vector().assign(ll);

        self
    }

    pub fn construct(self) -> WalkerState {
        self.state
    }
}

pub fn make_test_static_buffer() -> StaticBuffer {
    let mut buffer = StaticBuffer::new(2, 4, 2);

    buffer.record_state(&array![
        [11.0, 12.0],
        [13.0, 14.0],
        [15.0, 16.0],
        [17.0, 18.0]
    ]);
    buffer.record_state(&array![
        [21.0, 22.0],
        [23.0, 24.0],
        [25.0, 26.0],
        [27.0, 28.0]
    ]);

    buffer
}

pub struct TestNegativeAbsLogL {}

impl LogLikelihoodModel for TestNegativeAbsLogL {
    fn log_likelihood(&self, parameters: &ndarray::prelude::Array1<f64>) -> anyhow::Result<f64> {
        Ok(-parameters[0].abs())
    }
}

#[cfg(test)]
pub mod helpers {

    use ndarray::array;

    use crate::{
        chains::static_buffer::StaticBuffer, ess::EnsembleSliceSampler,
        log_likelihood::gaussian_1d_data_err::GaussianLl1dDataErrors,
    };

    // const MAX_N_STEPS: usize = 2;

    pub fn build_test_ess() -> EnsembleSliceSampler<StaticBuffer, GaussianLl1dDataErrors> {
        let data = array![[0.0_f64, 0.0_f64]];

        let chains = StaticBuffer::new(2, 4, 2);
        let model = GaussianLl1dDataErrors::new(data);

        let ess = EnsembleSliceSampler::<StaticBuffer, GaussianLl1dDataErrors>::new(
            2, 4, 2, chains, model,
        );

        ess
    }
}
