//! Builders for crate testing only

use ndarray::{Array1, Array2, array};

use crate::{chains::static_buffer::StaticBuffer, state::WalkerState};

/// Builder for a walker state object
pub struct WalkerStateBuilder {
    state: WalkerState,
}

impl WalkerStateBuilder {
    /// Create a new WalkerSpace instance with desired shape
    pub fn new(n_walkers: usize, n_parameters: usize) -> Self {
        Self {
            state: WalkerState::new(n_walkers, n_parameters),
        }
    }

    /// Assign state matrix
    pub fn set_state_matrix(mut self, x: Array2<f64>) -> Self {
        self.state.assign_state_matrix(x.view());

        self
    }

    /// Assign log-likelihood vector
    pub fn set_ll_vector(mut self, ll: Array1<f64>) -> Self {
        self.state.assign_ll_vector(ll.view());

        self
    }

    /// Switch 'current' <-> 'next' to enable filling both state matrices
    pub fn swap(mut self) -> Self {
        self.state.swap();

        self
    }

    /// Construct WalkerState object
    pub fn construct(self) -> WalkerState {
        self.state
    }
}

/// Generate a consistent StaticBuffer
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
