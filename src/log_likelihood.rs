pub mod gaussian_1d_data_err;

use ndarray::ArrayView1;

/// Trait for valid log-likelihood models.
pub trait LogLikelihoodModel {
    /// Returns the log-likelihood for a given set of parameters.
    fn log_likelihood(&self, parameters: ArrayView1<f64>) -> f64;
}
