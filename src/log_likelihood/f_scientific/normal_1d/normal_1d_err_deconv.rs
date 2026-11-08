//! Log-likelihood function

use ndarray::{Array1, Array2, ArrayView1}; //, ArrayView2, ArrayViewMut1, ArrayViewMut2};

use crate::{
    functions::likelihood::log_normalised_gaussian_s2,
    log_likelihood::{LogLikelihoodModel, ParameterTransformerConfig, Parametrised},
};

/// Normal distribution in one-dimension; error deconvolution
///
/// # Fields:
/// * `mu`: the position of the data points being fit
/// * `sigma_sqr`: the variance (standard deviation squared) of the data points being fit
pub struct GaussianLl1dDataErrors {
    values: Array1<f64>,
    errors_sqr: Array1<f64>,

    softplus_idx: Vec<usize>,
}

impl GaussianLl1dDataErrors {
    /// Constructor method
    ///
    /// Takes data in a columnwise format:
    /// data[:, 0], first column, corresponds to `mu`
    /// data[:, 1], second column, corresponds to `sigma` and will be converted to `sigma_sqr` internally
    pub fn new(data: &Array2<f64>) -> Self {
        let values = data.column(0).to_owned();
        let mut errors_sqr = data.column(1).to_owned();
        errors_sqr.iter_mut().for_each(|x| *x = *x * *x);

        let softplus_idx = vec![1];

        Self {
            values,
            errors_sqr,
            softplus_idx,
        }
    }

    /// Constructor method
    ///
    /// Takes individual arrays:
    /// * `mu`: array of the data values
    /// * `sigma`: array of data standard deviation; the conversion to `sigma_sqr` is handled internally
    pub fn from_values_errors(values: &Array1<f64>, errors: &Array1<f64>) -> Self {
        let values = values.to_owned();
        let mut errors_sqr = errors.to_owned();
        errors_sqr.iter_mut().for_each(|x| *x = *x * *x);

        let softplus_idx = vec![1];

        Self {
            values,
            errors_sqr,
            softplus_idx,
        }
    }
}

impl Parametrised for GaussianLl1dDataErrors {
    fn get_n_parameters(&self) -> usize {
        2
    }
}

impl ParameterTransformerConfig for GaussianLl1dDataErrors {
    const HAS_SOFTPLUS: bool = true;

    const HAS_STICK_BREAK: bool = false;
    const HAS_TANH: bool = false;
    const HAS_ORDSEQ: bool = false;
    const HAS_LOGISTIC: bool = false;

    fn softplus(&self) -> &[usize] {
        &self.softplus_idx
    }
}

impl LogLikelihoodModel for GaussianLl1dDataErrors {
    fn log_likelihood(&self, parameters: ArrayView1<f64>) -> anyhow::Result<f64> {
        let mu = parameters[0];
        let sigma = parameters[1];
        let sigma2 = sigma * sigma;

        let ll = self
            .values
            .iter()
            .zip(self.errors_sqr.iter())
            .map(|(v, e2)| log_normalised_gaussian_s2(*v, mu, sigma2 + e2))
            .sum::<f64>();

        Ok(ll)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::array;

    use super::*;
    use crate::{
        functions::transforms::softplus::softplus_inverse,
        log_likelihood::WrappedLogLikelihoodModel,
    };

    #[test]
    fn test_log_likelihood_computation() -> anyhow::Result<()> {
        let model = GaussianLl1dDataErrors::new(&array![[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]);

        // Parameters: mu = 2.0, log_sigma = ln(1.0) = 0.0
        let internal = array![2.0, softplus_inverse(1.0)?];
        let mut physical = array![0.0, 0.0];

        let ll = model.wrapped_log_likelihood(internal.view(), &mut physical.view_mut())?;

        assert_eq!(physical, array![2.0, 1.0]);

        // Expected: manually compute negative half sum of squared z-scores plus log term
        // z = (x - mu) / sigma = [-1, 0, 1], z^2 sum = 2
        // term1 = 2 * pi * sigma^2 = 2 * pi * 1 = 2pi
        // n * ln(term1) = 3 * ln(2pi)
        // log_likelihood = -0.5 * (3 ln(2pi) + 2)
        let softplus_term = softplus_inverse(1.0)? - 1.0; // this accounts for the Jacobian change of variables.
        let expected = softplus_term - 0.5 * (3.0 * (2.0 * std::f64::consts::PI).ln() + 2.0);

        assert_relative_eq!(ll, expected, epsilon = 1e-12);

        Ok(())
    }
}
