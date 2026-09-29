//! Log-likelihood function

use std::marker::PhantomData;

use ndarray::{Array1, Array2, ArrayView1};
use rand_distr::{Distribution, LogNormal, Normal};

use crate::{
    functions::{
        likelihood::log_normalised_gaussian_s2,
        transforms::{apply_transform_column, inv_softplus, lj_softplus_inplace, softplus},
    },
    log_likelihood::LogLikelihoodModel,
};

pub struct GaussianLl1dDataErrors {
    data: Array2<f64>,
    _type: PhantomData<f64>,
}

impl GaussianLl1dDataErrors {
    pub fn new(data: Array2<f64>) -> Self {
        Self {
            data,
            _type: PhantomData,
        }
    }

    fn get_data_values<'a>(&'a self) -> ArrayView1<'a, f64> {
        self.data.column(0)
    }

    fn get_data_errors<'a>(&'a self) -> ArrayView1<'a, f64> {
        self.data.column(1)
    }
}

impl LogLikelihoodModel for GaussianLl1dDataErrors {
    fn columnar_transform_physical_to_internal(&self, physical: &Array2<f64>) -> Array2<f64> {
        let mut internal = physical.to_owned();
        apply_transform_column(inv_softplus, &mut internal, 1);

        internal
    }

    fn columnar_transform_internal_to_physical(&self, internal: &Array2<f64>) -> Array2<f64> {
        let mut physical = internal.to_owned();
        apply_transform_column(softplus, &mut physical, 1);
        physical
    }

    fn convert_internal_to_physical_tracking_ll_warp(
        &self,
        parameters: ArrayView1<f64>,
    ) -> (f64, Array1<f64>) {
        let mut ll = 0.0;

        let mut modified_parameters = parameters.to_owned();

        lj_softplus_inplace(&mut modified_parameters[1], &mut ll);

        (ll, modified_parameters)
    }
    fn log_likelihood(&self, parameters: Array1<f64>) -> f64 {
        let mut ll = 0.0;

        let mu = parameters[0];
        let sigma2 = parameters[1].powi(2);

        let data_values = self.get_data_values();
        let data_errors = self.get_data_errors();

        ll += data_values
            .iter()
            .zip(data_errors.iter())
            .map(|(v, e)| {
                // get the expanded variance for use in the computation
                let sigma2conv = sigma2 + e.powi(2);

                log_normalised_gaussian_s2(*v, mu, sigma2conv)
            })
            .sum::<f64>();

        ll
    }
}

pub fn helper_generate_random_gaussian_points(n: usize) -> Array2<f64> {
    let mut data = Vec::<f64>::with_capacity(n * 2);

    let normal_v = Normal::<f64>::new(0.10, 0.80).unwrap();
    let normal_e = LogNormal::<f64>::new(0.00, 0.10).unwrap();

    let sampler = Normal::<f64>::new(0.00, 1.00).unwrap();

    let mut rng = rand::rng();

    for _ in 0..n {
        let mut value = normal_v.sample(&mut rng);
        let sigma = normal_e.sample(&mut rng);

        // this adds convolution with the error into the final mean value
        value += sigma * sampler.sample(&mut rng);

        data.push(value);
        data.push(sigma);
    }

    Array2::from_shape_vec((n, 2), data).unwrap()
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::array;

    use super::*;
    use crate::functions::transforms::inv_softplus;

    #[test]
    fn test_log_likelihood_computation() {
        let model = GaussianLl1dDataErrors::new(array![[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]);

        // Parameters: mu = 2.0, log_sigma = ln(1.0) = 0.0
        let parameters = array![2.0, inv_softplus(1.0)];

        let ll = model.internal_log_likelihood(parameters.view());

        // Expected: manually compute negative half sum of squared z-scores plus log term
        // z = (x - mu) / sigma = [-1, 0, 1], z^2 sum = 2
        // term1 = 2 * pi * sigma^2 = 2 * pi * 1 = 2pi
        // n * ln(term1) = 3 * ln(2pi)
        // log_likelihood = -0.5 * (3 ln(2pi) + 2)
        let softplus_term = inv_softplus(1.0) - 1.0; // this accounts for the Jacobian change of variables.
        let expected = softplus_term - 0.5 * (3.0 * (2.0 * std::f64::consts::PI).ln() + 2.0);

        assert_relative_eq!(ll, expected, epsilon = 1e-12);
    }

    #[test]
    fn test_helper_generate_random_gaussian_points_shape() {
        let n = 100;
        let arr = helper_generate_random_gaussian_points(n);
        assert_eq!(arr.shape(), &[n, 2]);
    }

    #[test]
    fn test_helper_generate_random_gaussian_points_is_finite() {
        let n = 100;
        let arr = helper_generate_random_gaussian_points(n);
        assert!(arr.iter().all(|x| x.is_finite()));
    }
}
