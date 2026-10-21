use anyhow;
use ndarray::ArrayView1;

use crate::log_likelihood::LogLikelihoodModel;

pub struct AnalyticRosenbrockToyLogL {
    a: f64,
    b: f64,
}

impl AnalyticRosenbrockToyLogL {
    pub fn new(a: f64, b: f64) -> Self {
        Self { a, b }
    }
}

impl LogLikelihoodModel for AnalyticRosenbrockToyLogL {
    fn log_likelihood(&self, parameters: ArrayView1<f64>) -> anyhow::Result<f64> {
        let x = parameters[0];
        let y = parameters[1];

        let ll = -0.5 * (self.a - x).powi(2) - self.b / 2.0 * (y - x.powi(2)).powi(2);

        Ok(ll)
    }
}
