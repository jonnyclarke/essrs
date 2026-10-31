//! Log-likelihood function

use anyhow;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

use crate::{
    functions::{
        likelihood::log_normalised_gaussian_s2,
        transforms::{
            apply_transform_column,
            softplus::{softplus, softplus_inverse, softplus_log_jacobian_inplace},
        },
    },
    log_likelihood::{LogLikelihoodModel, ProbThetaGivenDataModel},
};

pub struct GaussianLl1dDataErrors {
    mu: Array1<f64>,
    sigma_sqr: Array1<f64>,
}

impl GaussianLl1dDataErrors {
    pub fn new(data: &Array2<f64>) -> Self {
        let mu = data.column(0).to_owned();
        let mut sigma_sqr = data.column(1).to_owned();
        sigma_sqr.iter_mut().for_each(|x| *x = *x * *x);

        Self { mu, sigma_sqr }
    }

    pub fn from_mu_sigma(mu: &Array1<f64>, sigma: &Array1<f64>) -> Self {
        let mu = mu.to_owned();
        let mut sigma_sqr = sigma.to_owned();
        sigma_sqr.iter_mut().for_each(|x| *x = *x * *x);

        Self { mu, sigma_sqr }
    }
}

impl LogLikelihoodModel for GaussianLl1dDataErrors {
    fn get_n_parameters(&self) -> usize {
        2
    }

    fn columnar_transform_physical_to_internal(
        &self,
        physical: ArrayView2<f64>,
        mut internal: ArrayViewMut2<f64>,
    ) -> anyhow::Result<()> {
        internal.assign(&physical);
        apply_transform_column(softplus_inverse, &mut internal, 1)?;

        Ok(())
    }

    fn columnar_transform_internal_to_physical(
        &self,
        internal: ArrayView2<f64>,
        mut physical: ArrayViewMut2<f64>,
    ) -> anyhow::Result<()> {
        physical.assign(&internal);
        apply_transform_column(softplus, &mut physical, 1)?;

        Ok(())
    }

    fn convert_internal_to_physical_tracking_ll_warp(
        &self,
        log_jacobian: &mut f64,
        internal: ArrayView1<f64>,
        physical: &mut ArrayViewMut1<f64>,
    ) -> anyhow::Result<()> {
        physical.assign(&internal);

        softplus_log_jacobian_inplace(&mut physical[1], log_jacobian)?;

        Ok(())
    }

    fn log_likelihood(&self, parameters: ArrayView1<f64>) -> anyhow::Result<f64> {
        let mu = parameters[0];
        let sigma2 = parameters[1].powi(2);

        let ll = self
            .mu
            .iter()
            .zip(self.sigma_sqr.iter())
            .map(|(v, e2)| log_normalised_gaussian_s2(*v, mu, sigma2 + e2))
            .sum::<f64>();

        Ok(ll)
    }
}

impl ProbThetaGivenDataModel for GaussianLl1dDataErrors {
    fn log_likelihood_pointwise(&self, parameters: ArrayView1<f64>) -> anyhow::Result<Array1<f64>> {
        let mu = parameters[0];
        let sigma2 = parameters[1].powi(2);

        let ll = Array1::from_iter(
            self.mu
                .iter()
                .zip(self.sigma_sqr.iter())
                .map(|(v, e2)| log_normalised_gaussian_s2(*v, mu, sigma2 + e2)),
        );

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
