use anyhow;
use ndarray::ArrayView1;

use crate::log_likelihood::{
    LogLikelihoodModel, ParameterTransformer, Parametrised, ToyModelSamplerComponent,
};

const N_PARAMETERS_ROSENBROCK: usize = 2;

/// Rosenbrock toy model
///
/// # Fields:
/// * a: offset term
/// * b: shape term
pub struct AnalyticRosenbrockToyLogL {
    a: f64,
    b: f64,
}

impl AnalyticRosenbrockToyLogL {
    /// Constructor
    ///
    /// # Arguments:
    /// * a
    /// * b
    pub fn new(a: f64, b: f64) -> Self {
        Self { a, b }
    }
}

impl Parametrised for AnalyticRosenbrockToyLogL {
    fn get_n_parameters(&self) -> usize {
        N_PARAMETERS_ROSENBROCK
    }
}

impl ParameterTransformer for AnalyticRosenbrockToyLogL {}

impl LogLikelihoodModel for AnalyticRosenbrockToyLogL {
    fn log_likelihood(&self, parameters: ArrayView1<f64>) -> anyhow::Result<f64> {
        let x = parameters[0];
        let y = parameters[1];

        let ll = -0.5 * (self.a - x).powi(2) - self.b / 2.0 * (y - x.powi(2)).powi(2);

        Ok(ll)
    }
}

impl ToyModelSamplerComponent for AnalyticRosenbrockToyLogL {}

#[cfg(test)]
mod tests {

    use ndarray::array;

    use super::*;

    #[test]
    fn test_initialisation() -> anyhow::Result<()> {
        let ll_rosenbrock = AnalyticRosenbrockToyLogL::new(1.0, 3.0);
        let parameters = array![1.0, 1.0];
        let _ = ll_rosenbrock.log_likelihood(parameters.view());

        Ok(())
    }

    #[test]
    fn test_get_n_parameters() -> anyhow::Result<()> {
        let ll_rosenbrock = AnalyticRosenbrockToyLogL::new(1.0, 3.0);

        assert_eq!(ll_rosenbrock.get_n_parameters(), N_PARAMETERS_ROSENBROCK);

        Ok(())
    }
}
