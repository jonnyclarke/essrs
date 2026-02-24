pub mod gaussian_1d_data_err;

use ndarray::ArrayView1;

pub trait LogLikelihoodModel {
    fn log_likelihood(&self, parameters: ArrayView1<f64>) -> f64;
}
