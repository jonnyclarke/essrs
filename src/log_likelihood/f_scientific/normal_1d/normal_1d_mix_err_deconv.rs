//! Implementation of statistical uncertainty deconvolution

use anyhow::ensure;
use ndarray::{Array1, ArrayView1};

use crate::{
    functions::{likelihood::log_normalised_gaussian_s2, numerical::log_sum_exp},
    log_likelihood::{LogLikelihoodModel, ParameterTransformerConfig, Parametrised},
};

pub const N_PARAMETERS_NORMAL_1D: usize = 2;

/// Gaussian Mixture model
/// 2 Dimensions
/// Uncertainty De-Convolution
///
/// This requires 5 parameters per component:
/// * `mu_x`
/// * `mu_y`
/// * `sigma_x`
/// * `sigma_y`
/// * `p`
///
/// There is also an additional `n-1` parameters where n is the number of components which define the mixture weights.
///
/// Note:
/// - the sigmas must be >0
/// - the correlation coefficient must be `[-1 <= p <= 1]`
/// - the mixture weights must all be between `[0, 1]` and should sum to <1 (the final weight is equal to `1 - sum(w_i))`
pub struct Normal1DimMixErrDeconv {
    values: Array1<f64>,
    errors_sqr: Array1<f64>,

    n_components: usize,

    softplus: Vec<usize>,
    logistic: Vec<usize>,
    ordinal: Vec<usize>,
    stick_break: Vec<usize>,
}

#[allow(dead_code)]
impl Normal1DimMixErrDeconv {
    /// Constructor method
    ///
    /// # Arguments
    /// * `data`: Input of shape (n, 2) where the columns correspond to `x`, `err`
    /// * `n_components`: The number of components being fit in the mixture model
    ///
    /// # Errors
    /// * the number of components is < 1
    /// * length of value and errors match
    pub fn new(
        values: Array1<f64>,
        errors: Array1<f64>,
        n_components: usize,
    ) -> anyhow::Result<Self> {
        ensure!(
            n_components >= 2,
            "n components, {n_components}, should exceed 1"
        );
        ensure!(
            values.len().eq(&errors.len()),
            "lengths of x, ={:?}, is incompatible with length of error, ={:?}",
            values.len(),
            errors.len()
        );

        let mut ordinal = vec![];
        let mut softplus = vec![];
        let mut logistic = vec![];
        let mut stick_break = vec![];

        for i in 0..n_components - 1 {
            logistic.push(i);

            if n_components.ge(&3_usize) {
                stick_break.push(i)
            }
        }

        let k = n_components - 1;

        for i in 0..n_components {
            if n_components.ge(&2_usize) {
                let i_mu_x = k + i * N_PARAMETERS_NORMAL_1D;
                ordinal.push(i_mu_x); // mean x  
                if i.ge(&1_usize) {
                    softplus.push(i_mu_x)
                }
            }
            softplus.push(k + i * N_PARAMETERS_NORMAL_1D + 1); // sigma x
        }
        Ok(Self {
            values,
            errors_sqr: errors.iter().map(|x| x.powi(2)).collect(),
            n_components,
            ordinal,
            softplus,
            logistic,
            stick_break,
        })
    }

    fn get_n_components(&self) -> usize {
        self.n_components
    }

    // /// We need to ensure we account for the theta values
    // fn kth_offset(&self, k: usize) -> usize {
    //     self.n_components - 1 + k * N_PARAMETERS_NORMAL_1D
    // }

    // fn get_index_kth_x_mu(&self, k: usize) -> usize {
    //     self.kth_offset(k)
    // }

    // fn get_index_kth_y_mu(&self, k: usize) -> usize {
    //     self.kth_offset(k) + 1
    // }

    // fn get_index_kth_x_er(&self, k: usize) -> usize {
    //     self.kth_offset(k) + 2
    // }

    // fn get_index_kth_y_er(&self, k: usize) -> usize {
    //     self.kth_offset(k) + 3
    // }

    // fn get_index_kth_p(&self, k: usize) -> usize {
    //     self.kth_offset(k) + 4
    // }
}

impl Parametrised for Normal1DimMixErrDeconv {
    fn get_n_parameters(&self) -> usize {
        let n_thetas = self.get_n_components() - 1;
        let n_component_paramneters = self.get_n_components() * N_PARAMETERS_NORMAL_1D;

        n_thetas + n_component_paramneters
    }
}

impl ParameterTransformerConfig for Normal1DimMixErrDeconv {
    const HAS_SOFTPLUS: bool = true;
    const HAS_LOGISTIC: bool = true;
    const HAS_ORDSEQ: bool = true;
    const HAS_STICK_BREAK: bool = true;

    const HAS_TANH: bool = false;

    fn softplus(&self) -> &[usize] {
        &self.softplus
    }

    fn logistic(&self) -> &[usize] {
        &self.logistic
    }

    fn ordseq(&self) -> &[usize] {
        &self.ordinal
    }

    fn stick_break(&self) -> &[usize] {
        &self.stick_break
    }
}

impl LogLikelihoodModel for Normal1DimMixErrDeconv {
    /// here we have the annoying problem that we need to add raw likelihood together for each of the components.
    /// This requires use of the `log_sum_exp` function implemented in functions module.
    fn log_likelihood(&self, physical: ArrayView1<f64>) -> anyhow::Result<f64> {
        let n_data = self.values.len();
        let n_comp = self.get_n_components();

        let mut means = Vec::with_capacity(n_comp);
        let mut variances = Vec::with_capacity(n_comp);
        let mut log_weights = Vec::with_capacity(n_comp);

        for k in 0..n_comp {
            let j = n_comp - 1 + 2 * k;

            let mu = physical[j];
            let sigma = physical[j + 1];

            means.push(mu);
            variances.push(sigma * sigma);

            let theta = if k == n_comp - 1 {
                1.0_f64 - &log_weights.iter().map(|w: &f64| w.exp()).sum::<f64>()
            } else {
                physical[k]
            };

            log_weights.push(theta.ln());
        }

        let mut ll = 0.0_f64;

        for i in 0..n_data {
            // cache the data values
            let x = self.values[i];
            let sigma_sqr = self.errors_sqr[i];

            // initialise log-likelihood for data
            let mut ll_i = 0.0_f64;
            for (k, (&w, (&mu, &v))) in log_weights
                .iter()
                .zip(means.iter().zip(variances.iter()))
                .enumerate()
            {
                let ll_ik = w + log_normalised_gaussian_s2(x, mu, v + sigma_sqr);
                match k {
                    0 => ll_i = ll_ik,
                    _ => ll_i = log_sum_exp(ll_i, ll_ik)?,
                }
            }
            ll += ll_i;
        }

        Ok(ll)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Array2, array};
    use rstest::rstest;

    use super::*;
    use crate::{
        functions::{
            numerical::log_sum_exp,
            transforms::{logistic::logistic_inverse, softplus::softplus_inverse},
        },
        log_likelihood::ParameterTransformer,
    };

    #[rstest]
    #[case(
        array![1.1], array![1.2],
        2,
        vec![2, 3, 4],
        vec![0],
        vec![1, 3],
        vec![],
    )]
    #[case(
        array![1.1], array![1.2],
        3,
        vec![3,4,5,6,7],
        vec![0, 1],
        vec![2,4,6],
        vec![0, 1],
    )]
    fn test_gmm_builder(
        #[case] values: Array1<f64>,
        #[case] errors: Array1<f64>,
        #[case] n: usize,
        #[case] softplus: Vec<usize>,
        #[case] logistic: Vec<usize>,
        #[case] ordinal: Vec<usize>,
        #[case] stick_break: Vec<usize>,
    ) -> anyhow::Result<()> {
        let gmm = Normal1DimMixErrDeconv::new(values, errors, n)?;

        assert_eq!(gmm.softplus, softplus);
        assert_eq!(gmm.logistic, logistic);
        assert_eq!(gmm.ordinal, ordinal);
        assert_eq!(gmm.stick_break, stick_break);

        let _ = gmm.get_n_components();

        Ok(())
    }

    #[rstest]
    #[case(
        2,
        array![
            [
                0.5,
                1.1, 0.5,
                1.2, 0.4,
            ],
        ],
        array![
            [
                logistic_inverse(0.5)?,
                1.1, softplus_inverse(0.5)?,
                softplus_inverse(0.1)?, softplus_inverse(0.4)?
            ],
        ],
    )]
    fn test_columnar_transform_circularity(
        #[case] n_components: usize,
        #[case] physical: Array2<f64>,
        #[case] target_internal: Array2<f64>,
    ) -> anyhow::Result<()> {
        let gmm = Normal1DimMixErrDeconv::new(array![1.1], array![0.1], n_components)?;

        let mut internal = physical.clone();

        gmm.columnar_transform_physical_to_internal(physical.view(), internal.view_mut())?;

        assert_relative_eq!(internal, target_internal, epsilon = 1e-8);

        gmm.columnar_transform_internal_to_physical(target_internal.view(), internal.view_mut())?;

        assert_relative_eq!(internal, physical, epsilon = 1e-8);

        Ok(())
    }

    #[test]
    fn test_log_likelihood() -> anyhow::Result<()> {
        let gmm = Normal1DimMixErrDeconv::new(array![0.0], array![1.0], 2)?;

        let physical = array![0.4, 0.5, 1.2, -0.5, 1.5];

        // println!("true ll a: {:?}", 0.4_f64.ln() + log_normalised_gaussian_s2(0.0, 0.5, 1.2_f64.powi(2) + 1_f64));

        let ll = gmm.log_likelihood(physical.view())?;

        let true_ll = log_sum_exp(
            0.4_f64.ln() + log_normalised_gaussian_s2(0.0, 0.5, 1.2_f64.powi(2) + 1_f64),
            0.6_f64.ln() + log_normalised_gaussian_s2(0.0, -0.5, 1.5_f64.powi(2) + 1_f64),
        )?;

        assert_relative_eq!(true_ll, ll, epsilon = 1e-8);

        Ok(())
    }
}
