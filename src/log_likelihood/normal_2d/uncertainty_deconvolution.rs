//! Implementation of statistical uncertainty deconvolution

use anyhow;
use ndarray::{Array1, Array2, ArrayView1, Zip};

use crate::{
    functions::{
        likelihood::log_normal_2d,
        numerical::log_sum_exp,
        transforms::{
            apply_inplace_multi_column_transform, apply_transform_column,
            ordseq::{ordinal_sequencer, ordinal_sequencer_inverse},
            softplus::{softplus, softplus_inverse, softplus_log_jacobian_inplace},
            tanh::{tanh_inverse, tanh_log_jacobian_inplace, tanh_x},
        },
    },
    log_likelihood::LogLikelihoodModel,
};

pub const N_PARAMETERS_NORMAL_1D: usize = 5;

/// Gaussian Mixture model
/// 2 Dimensions
/// Uncertainty De-Convolution
///
/// This requires 5 parameters per component:
/// - [mu_x, mu_y], [sigma_x, sigma_y], [p]
///
/// Note:
/// - the sigmas must be >0
/// - the correlation coefficient must be [-1 <= p <= 1]
pub struct GaussianMixModel2DimUnDeConv {
    data: Array2<f64>,
    n_components: usize,
}

#[allow(dead_code)]
impl GaussianMixModel2DimUnDeConv {
    pub fn new(data: Array2<f64>, n_components: usize) -> Self {
        assert!(n_components >= 1, "N components must be greater than 1");
        assert!(data.ncols() == 5, "The data should have shape (n, 5)");
        Self { data, n_components }
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

    fn kth_offset(&self, k: usize) -> usize {
        k * N_PARAMETERS_NORMAL_1D
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

impl LogLikelihoodModel for GaussianMixModel2DimUnDeConv {
    fn columnar_transform_physical_to_internal(
        &self,
        physical: &Array2<f64>,
    ) -> anyhow::Result<Array2<f64>> {
        let n_comp = self.get_n_components();
        let mut internal = physical.to_owned();

        for k in 0..n_comp {
            apply_transform_column(softplus_inverse, &mut internal, self.get_index_kth_x_er(k))?;
            apply_transform_column(softplus_inverse, &mut internal, self.get_index_kth_y_er(k))?;
            apply_transform_column(tanh_inverse, &mut internal, self.get_index_kth_p(k))?;
        }

        for k in (0..n_comp).rev() {
            // we apply softplus inverse to sample not in linear space but mu_i + k space
            match k {
                0 => (),
                _ => {
                    // first we take difference to previous value
                    apply_inplace_multi_column_transform(
                        ordinal_sequencer_inverse,
                        &mut internal,
                        self.get_index_kth_x_mu(k - 1), // this value is subtracted
                        self.get_index_kth_x_mu(k),     // this value is modified
                    )?;
                    apply_transform_column(
                        softplus_inverse,
                        &mut internal,
                        self.get_index_kth_x_mu(k),
                    )?
                }
            }
        }

        Ok(internal)
    }

    fn columnar_transform_internal_to_physical(
        &self,
        internal: &Array2<f64>,
    ) -> anyhow::Result<Array2<f64>> {
        let n_comp = self.get_n_components();
        let mut physical = internal.to_owned();

        for k in 0..n_comp {
            // for all ordinal mean values, except anchor point, we first apply softplus transform to map unconstrained value to >0
            // we then add it to the (k-1)^th value
            match k {
                0 => (),
                _ => {
                    apply_transform_column(softplus, &mut physical, self.get_index_kth_x_mu(k))?;
                    apply_inplace_multi_column_transform(
                        ordinal_sequencer,
                        &mut physical,
                        self.get_index_kth_x_mu(k - 1),
                        self.get_index_kth_x_mu(k),
                    )?;
                }
            }
            apply_transform_column(softplus, &mut physical, self.get_index_kth_x_er(k))?;
            apply_transform_column(softplus, &mut physical, self.get_index_kth_y_er(k))?;
            apply_transform_column(tanh_x, &mut physical, self.get_index_kth_p(k))?;
        }

        Ok(physical)
    }

    fn convert_internal_to_physical_tracking_ll_warp(
        &self,
        internal: ArrayView1<f64>,
    ) -> anyhow::Result<(f64, Array1<f64>)> {
        let n_comp = self.get_n_components();

        let mut lj = 0.0;
        let mut physical = internal.to_owned();

        //         let i0 = self.get_index_kth_x_mu(k - 1);
        // let i1 = self.get_index_kth_x_mu(k);

        // let [x0, x1] = physical.get_many_mut([i0, i1])
        //     .expect("indices must be distinct");

        // ordinal_sequencer(&*x0, x1)?;

        for k in 0..n_comp {
            match k {
                0 => (),
                _ => {
                    softplus_log_jacobian_inplace(
                        &mut physical[self.get_index_kth_x_mu(k)],
                        &mut lj,
                    )?;
                    let x0 = physical[self.get_index_kth_x_mu(k - 1)];
                    ordinal_sequencer(x0, &mut physical[self.get_index_kth_x_mu(k)])?
                }
            }
            softplus_log_jacobian_inplace(&mut physical[self.get_index_kth_x_er(k)], &mut lj)?;
            softplus_log_jacobian_inplace(&mut physical[self.get_index_kth_y_er(k)], &mut lj)?;
            tanh_log_jacobian_inplace(&mut physical[self.get_index_kth_p(k)], &mut lj)?;
        }

        Ok((lj, physical))
    }

    /// here we have the annoying problem that we need to add raw likelihood together for each of the components.
    /// This requires use of the log_sum_exp function implemented in functions module.
    fn log_likelihood(&self, physical: &Array1<f64>) -> anyhow::Result<f64> {
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

        // this array will aggregate all of the contributions from separate gaussians together
        let mut ll_i = Array1::<f64>::zeros(n_data);

        for k in 0..n_comp {
            for i in 0..n_data {
                // logic for 2d normal log-likelihood
                let ll_ik = log_normal_2d(
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
                    0 => ll_i[i] = ll_ik,
                    _ => ll_i[i] = log_sum_exp(ll_i[i], ll_ik)?,
                }
            }
        }

        Ok(ll_i.sum())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    #[test]
    fn test_gmm_builder() {
        let gmm = GaussianMixModel2DimUnDeConv::new(array![[1.1, 1.2, 0.1, 0.2, 0.0]], 1);

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
    }
}
