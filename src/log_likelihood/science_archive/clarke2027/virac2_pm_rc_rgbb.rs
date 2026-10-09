/// Log Likelihood function to compute the join distribution
/// of the background RGBC component and the RC+B component
use ndarray::{Array1, Array2, ArrayView1, s};

use crate::functions::{
    likelihood::log_normalised_gaussian_s2, // version where sigma is passed already squared
    priors::ln_prior_normal,
};
use crate::{
    functions::{
        numerical::log_sum_exp,
        transforms::{
            apply_inplace_multi_column_transform, apply_transform_column,
            logistic::{logistic, logistic_inverse, logistic_log_jacobian_inplace},
            ordseq::{ordinal_sequencer, ordinal_sequencer_inverse},
            softplus::{softplus, softplus_inverse, softplus_log_jacobian_inplace},
        },
    },
    log_likelihood::LogLikelihoodModel,
};

pub struct C2027LlViracV2ProperMotionRcRgbb {
    b_coeffs: Array2<f64>,
    pm: Array1<f64>,
    pme2: Array1<f64>,
    ln_theta_rc: Array1<f64>,
    ln_one_sub_theta_rc: Array1<f64>,
    weight: Array1<f64>,
    n_nodes: usize,
}

impl C2027LlViracV2ProperMotionRcRgbb {
    pub fn new(
        b_coeffs: Array2<f64>,
        pm: Array1<f64>,
        pme: Array1<f64>,
        theta_rc: Array1<f64>,
        weight: Array1<f64>,
    ) -> Self {
        let n_nodes = b_coeffs.shape()[1];

        assert!(pm.shape()[0] == pme.shape()[0]);
        assert!(pm.shape()[0] == b_coeffs.shape()[0]);

        let ln_theta_rc = theta_rc.iter().map(|x| x.ln()).collect();
        let ln_one_sub_theta_rc = theta_rc.iter().map(|x| (1.0 - x).ln()).collect();
        Self {
            b_coeffs,
            pm,
            pme2: pme.iter().map(|x| x.powi(2)).collect(),
            ln_theta_rc,
            ln_one_sub_theta_rc,
            weight,
            n_nodes,
        }
    }
}

impl LogLikelihoodModel for C2027LlViracV2ProperMotionRcRgbb {
    fn columnar_transform_physical_to_internal(
        &self,
        physical: ndarray::prelude::ArrayView2<f64>,
        mut internal: ndarray::prelude::ArrayViewMut2<f64>,
    ) -> anyhow::Result<()> {
        internal.assign(&physical);

        // the sigma nodes
        for k in self.n_nodes..2 * self.n_nodes {
            apply_transform_column(softplus_inverse, &mut internal, k)?;
        }

        let i = 2 * self.n_nodes;

        // Gaussian Mixture Background
        apply_transform_column(logistic_inverse, &mut internal, i)?;

        apply_inplace_multi_column_transform(
            ordinal_sequencer_inverse,
            &mut internal,
            i + 1,
            i + 2,
        )?;
        apply_transform_column(softplus_inverse, &mut internal, i + 2)?;

        apply_transform_column(softplus_inverse, &mut internal, i + 3)?;
        apply_transform_column(softplus_inverse, &mut internal, i + 4)?;

        Ok(())
    }

    fn columnar_transform_internal_to_physical(
        &self,
        internal: ndarray::prelude::ArrayView2<f64>,
        mut physical: ndarray::prelude::ArrayViewMut2<f64>,
    ) -> anyhow::Result<()> {
        physical.assign(&internal);

        // the sigma nodes
        for k in self.n_nodes..2 * self.n_nodes {
            apply_transform_column(softplus, &mut physical, k)?;
        }

        let i = 2 * self.n_nodes;

        // Gaussian Mixture Background
        apply_transform_column(logistic, &mut physical, i)?;

        apply_transform_column(softplus, &mut physical, i + 2)?;
        apply_inplace_multi_column_transform(ordinal_sequencer, &mut physical, i + 1, i + 2)?;

        apply_transform_column(softplus, &mut physical, i + 3)?;
        apply_transform_column(softplus, &mut physical, i + 4)?;

        Ok(())
    }

    fn convert_internal_to_physical_tracking_ll_warp(
        &self,
        log_jacobian: &mut f64,
        internal: ArrayView1<f64>,
        physical: &mut ndarray::prelude::ArrayViewMut1<f64>,
    ) -> anyhow::Result<()> {
        physical.assign(&internal);

        // the sigma nodes
        for k in self.n_nodes..2 * self.n_nodes {
            softplus_log_jacobian_inplace(&mut physical[k], log_jacobian)?;
        }

        let i = 2 * self.n_nodes;

        // Gaussian Mixture Background
        logistic_log_jacobian_inplace(&mut physical[i], log_jacobian)?;

        softplus_log_jacobian_inplace(&mut physical[i + 2], log_jacobian)?;
        ordinal_sequencer(physical[i + 1], &mut physical[i + 2])?;

        softplus_log_jacobian_inplace(&mut physical[i + 3], log_jacobian)?;
        softplus_log_jacobian_inplace(&mut physical[i + 4], log_jacobian)?;

        Ok(())
    }

    fn log_likelihood(&self, parameters: ArrayView1<f64>) -> anyhow::Result<f64> {
        let mu_nodes = parameters.slice(s![0..self.n_nodes]);
        let sig_nodes = parameters.slice(s![self.n_nodes..2 * self.n_nodes]);

        let i = 2 * self.n_nodes;
        // let tau_mu = parameters[i]; // hierarchical priors -- mu smoothing
        // let tau_sigma = parameters[i + 1]; // hierarchical priors -- sigma smoothing

        // parameters of the background gaussian mixture model
        let phi_rgbc = parameters[i];
        let mu0_rgbc = parameters[i + 1];
        let mu1_rgbc = parameters[i + 2];

        // this is already squared for adding in quadrature
        let sigma0_rgbc = parameters[i + 3].powi(2);
        let sigma1_rgbc = parameters[i + 4].powi(2);

        let ln_phi_rgbc = phi_rgbc.ln();
        let ln_one_sub_phi_rgbc = (1.0 - phi_rgbc).ln();

        let mut ll = 0.0;

        // this is to try and prevent node explosion...
        for i in 0..self.n_nodes {
            ln_prior_normal(mu_nodes[i], -5.0, 5.0, &mut ll);
            ln_prior_normal(sig_nodes[i], 3.0, 5.0, &mut ll);
        }

        for (i, (x, e2)) in self.pm.iter().zip(self.pme2.iter()).enumerate() {
            let pred_mu = self
                .b_coeffs
                .row(i)
                .iter()
                .zip(mu_nodes.iter())
                .map(|(&b, &n)| b * n)
                .sum::<f64>();

            let pred_sigma = self
                .b_coeffs
                .row(i)
                .iter()
                .zip(sig_nodes.iter())
                .map(|(&b, &n)| b * n)
                .sum::<f64>();

            if pred_sigma <= 0.0 {
                return Ok(f64::NEG_INFINITY);
            }

            ll += self.weight[i]
                * log_sum_exp(
                    // Red clump component
                    self.ln_theta_rc[i]
                        + log_normalised_gaussian_s2(*x, pred_mu, *e2 + pred_sigma.powi(2)),
                    // RGBC component split into two possible sub-gaussians
                    self.ln_one_sub_theta_rc[i]
                        + log_sum_exp(
                            ln_phi_rgbc
                                + log_normalised_gaussian_s2(*x, mu0_rgbc, e2 + sigma0_rgbc),
                            ln_one_sub_phi_rgbc
                                + log_normalised_gaussian_s2(*x, mu1_rgbc, e2 + sigma1_rgbc),
                        )?,
                )?
        }

        Ok(ll)
    }
}
