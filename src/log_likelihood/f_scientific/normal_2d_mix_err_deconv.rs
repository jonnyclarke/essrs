//! Implementation of statistical uncertainty deconvolution

use anyhow::ensure;
use ndarray::{Array2, ArrayView1, Zip, s};

use crate::{
    functions::{likelihood::log_normal_2d, numerical::log_sum_exp},
    log_likelihood::{LogLikelihoodModel, ParameterTransformerConfig, Parametrised},
};

pub const N_PARAMETERS_NORMAL_2D: usize = 5;

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
pub struct GaussianMixModel2DimUnDeConv {
    data: Array2<f64>,
    n_components: usize,

    softplus: Vec<usize>,
    tanh: Vec<usize>,
    logistic: Vec<usize>,
    ordinal: Vec<usize>,
    stick_break: Vec<usize>,
}

#[allow(dead_code)]
impl GaussianMixModel2DimUnDeConv {
    /// Constructor method
    ///
    /// # Arguments
    /// * `data`: Input of shape (n, 5) where the columns correspond to `x`, `y`, `x_err`, `y_err`, `corr`
    /// * `n_components`: The number of components being fit in the mixture model
    ///
    /// # Errors
    /// * the number of components is < 1
    /// * the data is of the correct shape (should be [n, 5])
    pub fn new(data: Array2<f64>, n_components: usize) -> anyhow::Result<Self> {
        ensure!(
            n_components >= 1,
            "n components, {n_components}, should exceed 1"
        );
        ensure!(
            data.ncols() == 5,
            "the data, with shape [{}, {}], should have shape [n, 5]",
            data.nrows(),
            data.ncols()
        );

        let mut ordinal = vec![];
        let mut softplus = vec![];
        let mut tanh = vec![];
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
                let i_mu_x = k + i * N_PARAMETERS_NORMAL_2D;
                ordinal.push(i_mu_x); // mean x  
                if i.ge(&1_usize) {
                    softplus.push(i_mu_x)
                }
            }
            softplus.push(k + i * N_PARAMETERS_NORMAL_2D + 2); // sigma x
            softplus.push(k + i * N_PARAMETERS_NORMAL_2D + 3); // sigma y
            tanh.push(k + i * N_PARAMETERS_NORMAL_2D + 4);
        }
        Ok(Self {
            data,
            n_components,
            ordinal,
            softplus,
            tanh,
            logistic,
            stick_break,
        })
    }

    fn get_x_value<'a>(&'a self) -> ArrayView1<'a, f64> {
        self.data.column(0)
    }

    fn get_y_value<'a>(&'a self) -> ArrayView1<'a, f64> {
        self.data.column(1)
    }

    fn get_x_error<'a>(&'a self) -> ArrayView1<'a, f64> {
        self.data.column(2)
    }

    fn get_y_error<'a>(&'a self) -> ArrayView1<'a, f64> {
        self.data.column(3)
    }

    fn get_p<'a>(&'a self) -> ArrayView1<'a, f64> {
        self.data.column(4)
    }

    fn get_n_components(&self) -> usize {
        self.n_components
    }

    /// We need to ensure we account for the theta values
    fn kth_offset(&self, k: usize) -> usize {
        self.n_components - 1 + k * N_PARAMETERS_NORMAL_2D
    }

    fn get_index_kth_x_mu(&self, k: usize) -> usize {
        self.kth_offset(k)
    }

    fn get_index_kth_y_mu(&self, k: usize) -> usize {
        self.kth_offset(k) + 1
    }

    fn get_index_kth_x_er(&self, k: usize) -> usize {
        self.kth_offset(k) + 2
    }

    fn get_index_kth_y_er(&self, k: usize) -> usize {
        self.kth_offset(k) + 3
    }

    fn get_index_kth_p(&self, k: usize) -> usize {
        self.kth_offset(k) + 4
    }
}

impl Parametrised for GaussianMixModel2DimUnDeConv {
    fn get_n_parameters(&self) -> usize {
        let n_thetas = self.get_n_components() - 1;
        let n_component_paramneters = self.get_n_components() * N_PARAMETERS_NORMAL_2D;

        n_thetas + n_component_paramneters
    }
}

impl ParameterTransformerConfig for GaussianMixModel2DimUnDeConv {
    const HAS_LOGISTIC: bool = true;
    const HAS_SOFTPLUS: bool = true;
    const HAS_TANH: bool = true;
    const HAS_ORDSEQ: bool = true;
    const HAS_STICK_BREAK: bool = true;

    fn softplus(&self) -> &[usize] {
        &self.softplus
    }

    fn tanh(&self) -> &[usize] {
        &self.tanh
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

impl LogLikelihoodModel for GaussianMixModel2DimUnDeConv {
    /// here we have the annoying problem that we need to add raw likelihood together for each of the components.
    /// This requires use of the `log_sum_exp` function implemented in functions module.
    fn log_likelihood(&self, physical: ArrayView1<f64>) -> anyhow::Result<f64> {
        // extract views of the data
        let x = self.get_x_value();
        let y = self.get_y_value();

        let x_e = self.get_x_error();
        let y_e = self.get_y_error();

        let p = self.get_p();

        // now derive the values used in the normal 2d formula
        let a = x_e.mapv(|v| v.powi(2));
        let d = y_e.mapv(|v| v.powi(2));
        let mut bc = x_e.to_owned();
        Zip::from(&mut bc)
            .and(&y_e)
            .and(&p)
            .for_each(|bc, &y_e, &p| {
                *bc *= y_e * p;
            });

        let n_data = x.len();
        let n_comp = self.get_n_components();

        let mut thetas = physical.slice(s![0..self.n_components - 1]).to_vec();
        let last_theta = 1.0_f64 - thetas.iter().sum::<f64>();

        thetas.push(last_theta);

        let mut ll = 0.0_f64;

        for i in 0..n_data {
            let mut ll_i = 0.0_f64;
            for k in 0..n_comp {
                // logic for 2d normal log-likelihood
                let ll_ik = thetas[k].ln()
                    + log_normal_2d(
                        x[i],
                        y[i],
                        physical[self.get_index_kth_x_mu(k)],
                        physical[self.get_index_kth_y_mu(k)],
                        a[i] + physical[self.get_index_kth_x_er(k)].powi(2),
                        bc[i]
                            + (physical[self.get_index_kth_x_er(k)]
                                * physical[self.get_index_kth_y_er(k)]
                                * physical[self.get_index_kth_p(k)]),
                        d[i] + physical[self.get_index_kth_y_er(k)].powi(2),
                    );
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
    use ndarray::array;
    use rstest::rstest;

    use super::*;
    use crate::{
        functions::transforms::{
            logistic::logistic_inverse, softplus::softplus_inverse, tanh::tanh_inverse,
        },
        log_likelihood::ParameterTransformer,
    };

    #[rstest]
    #[case(
        array![[1.1_f64, 1.2, 0.1, 0.2, 0.0]],
        1,
        vec![2_usize, 3],
        vec![],
        vec![4_usize],
        vec![],
        vec![],
    )]
    #[case(
        array![[1.1_f64, 1.2, 0.1, 0.2, 0.0,]],
        2,
        vec![3, 4, 6, 8, 9],
        vec![0],
        vec![5_usize, 10],
        vec![1, 6],
        vec![],
    )]
    #[case(
        array![[1.1_f64, 1.2, 0.1, 0.2, 0.0,]],
        3,
        vec![4, 5, 7, 9, 10, 12, 14, 15],
        vec![0, 1],
        vec![6, 11, 16],
        vec![2, 7, 12],
        vec![0, 1],
    )]
    fn test_gmm_builder(
        #[case] a: Array2<f64>,
        #[case] n: usize,
        #[case] softplus: Vec<usize>,
        #[case] logistic: Vec<usize>,
        #[case] tanh: Vec<usize>,
        #[case] ordinal: Vec<usize>,
        #[case] stick_break: Vec<usize>,
    ) -> anyhow::Result<()> {
        let gmm = GaussianMixModel2DimUnDeConv::new(a, n)?;

        assert_eq!(gmm.softplus, softplus);
        assert_eq!(gmm.logistic, logistic);
        assert_eq!(gmm.tanh, tanh);
        assert_eq!(gmm.ordinal, ordinal);
        assert_eq!(gmm.stick_break, stick_break);

        let _ = gmm.get_x_value();
        let _ = gmm.get_y_value();
        let _ = gmm.get_x_error();
        let _ = gmm.get_y_error();
        let _ = gmm.get_p();

        let _ = gmm.get_n_components();
        let _ = gmm.get_index_kth_x_mu(0);
        let _ = gmm.get_index_kth_x_er(0);
        let _ = gmm.get_index_kth_y_mu(0);
        let _ = gmm.get_index_kth_y_er(0);
        let _ = gmm.get_index_kth_p(0);

        Ok(())
    }

    #[rstest]
    #[case(
        2,
        array![
            [
                0.5,
                1.1, 1.1, 0.5, 0.5, 0.2,
                1.2, 1.3, 0.5, 0.5, 0.2
            ],
        ],
        array![
            [
                logistic_inverse(0.5)?,
                1.1, 1.1, softplus_inverse(0.5)?, softplus_inverse(0.5)?, tanh_inverse(0.2)?,
                softplus_inverse(0.1)?, 1.3, softplus_inverse(0.5)?, softplus_inverse(0.5)?, tanh_inverse(0.2)?
            ],
        ],
    )]
    fn test_columnar_transform_circularity(
        #[case] n_components: usize,
        #[case] physical: Array2<f64>,
        #[case] target_internal: Array2<f64>,
    ) -> anyhow::Result<()> {
        let d = array![[1.1, 1.2, 0.1, 0.2, 0.0]];

        let gmm = GaussianMixModel2DimUnDeConv::new(d, n_components)?;

        let mut internal = physical.clone();

        gmm.columnar_transform_physical_to_internal(physical.view(), internal.view_mut())?;

        assert_relative_eq!(internal, target_internal, epsilon = 1e-8);

        gmm.columnar_transform_internal_to_physical(target_internal.view(), internal.view_mut())?;

        assert_relative_eq!(internal, physical, epsilon = 1e-8);

        Ok(())
    }
}
