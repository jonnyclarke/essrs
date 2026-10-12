/// Likelihood function to compute the mean of the overall velocity distribution
/// 
use ndarray::{
    Array1, ArrayView1,
    Array2,
    s
};

use crate::log_likelihood::LogLikelihoodModel;
use crate::functions::{
    likelihood::log_normalised_gaussian_s2,  // version where sigma is passed already squared
    priors::{
        ln_prior_exponential,
        ln_prior_normal,
    },
    transforms::softplus::{softplus_log_jacobian_inplace, softplus_inverse, softplus},
    transforms::apply_transform_column
};


pub struct C2027LlViracV2ProperMotion {
    b_coeffs: Array2<f64>,
    pml: Array1<f64>,
    pmle2: Array1<f64>,

    n_nodes: usize
}

impl C2027LlViracV2ProperMotion {
    pub fn new(
        b_coeffs: Array2<f64>,
        pml: Array1<f64>,
        pmle: Array1<f64>,
    ) -> Self {
        let n_nodes = b_coeffs.shape()[1];

        assert!(pml.shape()[0] == pmle.shape()[0]);
        assert!(pml.shape()[0] == b_coeffs.shape()[0]);

        Self {
            b_coeffs,
            pml,
            pmle2: pmle.iter().map(|x| x.powi(2)).collect(),
            n_nodes,
        }
    }
}

impl LogLikelihoodModel for C2027LlViracV2ProperMotion {

    fn columnar_transform_physical_to_internal(
        &self,
        physical: ndarray::prelude::ArrayView2<f64>,
        mut internal: ndarray::prelude::ArrayViewMut2<f64>,
    ) -> anyhow::Result<()>
    {
        internal.assign(&physical);

        // the sigma nodes
        for k in self.n_nodes..2*self.n_nodes {
            apply_transform_column(softplus_inverse, &mut internal, k)?;
        }

        let i = 2 * self.n_nodes;
        // hierarchical smoothing hyper-parameters
        apply_transform_column(softplus_inverse, &mut internal, i)?;
        apply_transform_column(softplus_inverse, &mut internal, i + 1)?;

        Ok(())
    }

    fn columnar_transform_internal_to_physical(
        &self,
        internal: ndarray::prelude::ArrayView2<f64>,
        mut physical: ndarray::prelude::ArrayViewMut2<f64>,
    ) -> anyhow::Result<()>
    {
        physical.assign(&internal);

        // the sigma nodes
        for k in self.n_nodes..2*self.n_nodes {
            apply_transform_column(softplus, &mut physical, k)?;
        }

        // hierarchical smoothing hyper-parameters
        apply_transform_column(softplus, &mut physical, 2*self.n_nodes)?;
        apply_transform_column(softplus, &mut physical, 2*self.n_nodes + 1)?;
        
        Ok(())
    }

    fn convert_internal_to_physical_tracking_ll_warp(
        &self,
        log_jacobian: &mut f64,
        internal: ArrayView1<f64>,
        physical: &mut ndarray::prelude::ArrayViewMut1<f64>,
    ) -> anyhow::Result<()>
    {
        physical.assign(&internal);

        // the sigma nodes
        for k in self.n_nodes..2*self.n_nodes {
            softplus_log_jacobian_inplace(&mut physical[k], log_jacobian)?;
        }

        // hierarchical smoothing hyper-parameters
        softplus_log_jacobian_inplace(&mut physical[2*self.n_nodes], log_jacobian)?;
        softplus_log_jacobian_inplace(&mut physical[2*self.n_nodes + 1], log_jacobian)?;

        Ok(())
    }

    fn log_likelihood(
            &self,
            parameters: ArrayView1<f64>,
        ) -> anyhow::Result<f64> {

            let mu_nodes = parameters.slice(s![0*self.n_nodes..1*self.n_nodes]);
            let sig_nodes = parameters.slice(s![1*self.n_nodes..2*self.n_nodes]);

            let tau_mu = parameters[self.n_nodes * 2];         // hierarchical priors -- mu smoothing
            let tau_sigma = parameters[self.n_nodes * 2 + 1];  // hierarchical priors -- sigma smoothing

            let mut ll = 0.0;

            // ensure that the hyper-parameters are drawn from the exponential distribution
            ln_prior_exponential(tau_mu, 1.0, &mut ll);
            ln_prior_exponential(tau_sigma, 1.0, &mut ll);

            // apply a prior on the smoothing node values
            for k in 2..self.n_nodes {
                ln_prior_normal(
                    mu_nodes[k] - 2.0 * mu_nodes[k-1] + mu_nodes[k-2],
                    0.0,
                    tau_mu,
                    &mut ll
                );
                ln_prior_normal(
                    sig_nodes[k] - 2.0 * sig_nodes[k-1] + sig_nodes[k-2],
                    0.0,
                    tau_sigma,
                    &mut ll
                );
            }

            for (i,(x, e2)) in self.pml.iter().zip(self.pmle2.iter()).enumerate() {

                let pred_mu = self.b_coeffs.row(i)
                    .iter()
                    .zip(
                        mu_nodes
                        .iter()
                    )
                    .map(|(&b, &n)| b * n)
                    .sum::<f64>();

                let pred_sigma = self.b_coeffs.row(i)
                    .iter()
                    .zip(
                        sig_nodes
                        .iter()
                    )
                    .map(|(&b, &n)| b * n)
                    .sum::<f64>();

                if pred_sigma <= 0.0 {
                    return Ok(f64::NEG_INFINITY)
                }

                ll += log_normalised_gaussian_s2(
                    *x,
                    pred_mu,
                    *e2 + pred_sigma.powi(2)
                )
            }

        Ok(ll)
    }
}
