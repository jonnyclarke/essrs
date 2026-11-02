//! Builders for crate testing only

use anyhow::ensure;
use ndarray::{Array1, Array2, ArrayView1, array};

use crate::{
    chains::static_buffer::StaticBuffer, log_likelihood::LogLikelihoodModel, state::WalkerState,
};

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

/// Testing log-likelihood implementing a uniform probability distribution
pub struct TestLeftRightUniformLogL {
    l: f64,
    r: f64,
    p: f64,
}

impl TestLeftRightUniformLogL {
    /// Constructor
    pub fn new(l: f64, r: f64) -> anyhow::Result<Self> {
        ensure!(
            r.gt(&l),
            "require l < r in uniform log likelihood, arguments are l(={l}) >= r(={r})"
        );
        let p = 1.0 / (r - l);
        Ok(Self { l, r, p })
    }

    /// Get the value of the left and right bounds of the uniform distribution
    pub fn get_lr(&self) -> (f64, f64) {
        (self.l, self.r)
    }

    /// Get the probability value corresponding to the uniform distribution (integral from left to right normalised to 1)
    pub fn get_p(&self) -> f64 {
        self.p
    }
}

impl Default for TestLeftRightUniformLogL {
    fn default() -> Self {
        Self::new(-5.0, 5.0).unwrap()
    }
}

impl LogLikelihoodModel for TestLeftRightUniformLogL {
    fn get_n_parameters(&self) -> usize {
        1
    }

    fn log_likelihood(&self, parameters: ArrayView1<f64>) -> anyhow::Result<f64> {
        if parameters[0].gt(&self.l) && parameters[0].lt(&self.r) {
            return Ok(self.p);
        }

        Ok(f64::NEG_INFINITY)
    }
}
