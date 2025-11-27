use std::marker::PhantomData;

use ndarray::Array1;

use crate::{
    FloatExt, data::DataBufferTrait, ess::EnsembleSliceSamplerConfigTrait,
    log_likelihood::LogLikelihoodModel,
};

pub struct GaussianLogLikelihood1D<T: FloatExt, CONFIG: EnsembleSliceSamplerConfigTrait> {
    _type: PhantomData<T>,
    _config: PhantomData<CONFIG>,
}

impl<T: FloatExt, CONFIG: EnsembleSliceSamplerConfigTrait> LogLikelihoodModel<T>
    for GaussianLogLikelihood1D<T, CONFIG>
{
    const N_PARAMETERS: usize = 2;
    const N_DATA_DIMENS: usize = 1; //TODO: make this a fitting with errors gaussian

    fn n_parameters(&self) -> usize {
        Self::N_PARAMETERS
    }

    fn log_likelihood<DataType: DataBufferTrait<T>>(
        &self,
        data: &DataType,
        parameters: &Array1<T>,
    ) -> T {
        let n: usize = data.n_data_points();

        let two = T::from(2.0).unwrap();
        let neg_half = T::from(-0.5).unwrap();
        let n_as_t: T = T::from(n).unwrap();

        let mu: T = parameters[0];
        let sigma: T = parameters[1].exp();

        let term1: T = two * T::my_pi() * sigma.powi(2);

        let data_mean: ndarray::ArrayBase<ndarray::ViewRepr<&T>, ndarray::Dim<[usize; 1]>, T> =
            data.get_dimension(0);

        let term2: T = data_mean
            .iter()
            .map(|x: &T| {
                let z: T = (*x - mu) / sigma;
                z * z
            })
            .sum();

        neg_half * (n_as_t * term1.ln() + term2)
    }

    fn jacobian_change_of_variables(&self, parameters: &Array1<T>) -> T {
        parameters[1]
    }
}

impl<T: FloatExt, CONFIG: EnsembleSliceSamplerConfigTrait> GaussianLogLikelihood1D<T, CONFIG> {
    const _RUNTIME_TIME_ASSERTION_DATA_DIMENS: () = assert!(
        Self::N_DATA_DIMENS == CONFIG::N_DATA_DIMENS,
        "CONFIG::N_PARAMETERS does not match MODEL::N_PARAMETERS"
    );

    pub fn new() -> Self {
        Self {
            _type: PhantomData,
            _config: PhantomData,
        }
    }
}

impl<T: FloatExt, CONFIG: EnsembleSliceSamplerConfigTrait> Default
    for GaussianLogLikelihood1D<T, CONFIG>
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::{data::DataBuffer, ess::EnsembleSliceSamplerConfig};

    // --- Mock Config ---
    type TestConfig = EnsembleSliceSamplerConfig<10, 2, 2, 1, 1, 1, true>;

    #[test]
    fn test_log_likelihood_computation() {
        let model = GaussianLogLikelihood1D::<f64, TestConfig>::default();

        // Fake data: 1D points
        let data = DataBuffer::<f64, TestConfig>::new(array![[1.0], [2.0], [3.0]]);

        // Parameters: mu = 2.0, log_sigma = ln(1.0) = 0.0
        let parameters = array![2.0, 0.0];

        let ll = model.log_likelihood(&data, &parameters);

        // Expected: manually compute negative half sum of squared z-scores plus log term
        // z = (x - mu) / sigma = [-1, 0, 1], z^2 sum = 2
        // term1 = 2 * pi * sigma^2 = 2 * pi * 1 = 2pi
        // n * ln(term1) = 3 * ln(2pi)
        // log_likelihood = -0.5 * (3 ln(2pi) + 2)
        let expected = -0.5 * (3.0 * (2.0 * std::f64::consts::PI).ln() + 2.0);

        let tol = 1e-12;
        assert!((ll - expected).abs() < tol, "Log-likelihood mismatch");
    }

    #[test]
    fn test_jacobian_change_of_variables() {
        let model = GaussianLogLikelihood1D::<f64, TestConfig>::new();
        let parameters = array![0.0, 0.5];

        assert_eq!(model.jacobian_change_of_variables(&parameters), 0.5);
    }

    #[test]
    fn test_n_parameters() {
        let model = GaussianLogLikelihood1D::<f64, TestConfig>::new();
        assert_eq!(model.n_parameters(), 2);
    }
}
