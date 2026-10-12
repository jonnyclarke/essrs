/// Log likelihood function for fitting VIRAC-2 density profiles with
/// an exponential and gaussian mixture
use ndarray::{Array1, ArrayView1};

use crate::{
    functions::{
        likelihood::log_normalised_gaussian,
        numerical::{log_diff_exp, log_sum_exp},
        priors::{ln_prior_exponential, ln_prior_normal},
        transforms::{
            apply_transform_column,
            logistic::{logistic, logistic_inverse, logistic_log_jacobian_inplace},
            softplus::{softplus, softplus_inverse, softplus_log_jacobian_inplace},
        },
    },
    log_likelihood::LogLikelihoodModel,
};

pub struct Clarke2027LogLiViracV2RcRgbb {
    ks: Array1<f64>,
    w_rc: Array1<f64>,
    ks_min: f64,
    ks_max: f64,
}

impl Clarke2027LogLiViracV2RcRgbb {
    pub fn new(ks: Array1<f64>, w_rc: Array1<f64>, ks_min: f64, ks_max: f64) -> Self {
        Self {
            ks,
            w_rc,
            ks_min,
            ks_max,
        }
    }
}

/// Parameters of this model:
/// - 0: theta   :: weight between the exponential and gaussian mixture
/// - 1: alpha   :: exponential scale height
/// - 2: mu      :: mean of RC Gaussian component
/// - 3: sigma   :: standard deviation of RC gaussian
/// - 4: d_mu    :: Delta central magnitude [RGBB - RC]
/// - 5: f_sigma :: Ratio of RGBB to RC dispersion [constrained >0]
/// - 6: w_rc    :: Weight of RC [RC_w + RGBB_w = 1]
///
/// Hyperparameters of the model
/// - 7: tau    :: parameter controlling prior on w_rc
impl LogLikelihoodModel for Clarke2027LogLiViracV2RcRgbb {
    fn columnar_transform_physical_to_internal(
        &self,
        physical: ndarray::prelude::ArrayView2<f64>,
        mut internal: ndarray::prelude::ArrayViewMut2<f64>,
    ) -> anyhow::Result<()> {
        internal.assign(&physical);

        apply_transform_column(logistic_inverse, &mut internal, 0)?;
        apply_transform_column(softplus_inverse, &mut internal, 1)?;
        apply_transform_column(softplus_inverse, &mut internal, 3)?;
        apply_transform_column(softplus_inverse, &mut internal, 4)?;
        apply_transform_column(softplus_inverse, &mut internal, 5)?;
        apply_transform_column(logistic_inverse, &mut internal, 6)?;

        apply_transform_column(softplus_inverse, &mut internal, 7)?;

        Ok(())
    }

    fn columnar_transform_internal_to_physical(
        &self,
        internal: ndarray::prelude::ArrayView2<f64>,
        mut physical: ndarray::prelude::ArrayViewMut2<f64>,
    ) -> anyhow::Result<()> {
        physical.assign(&internal);

        apply_transform_column(logistic, &mut physical, 0)?;
        apply_transform_column(softplus, &mut physical, 1)?;
        apply_transform_column(softplus, &mut physical, 3)?;
        apply_transform_column(softplus, &mut physical, 4)?;
        apply_transform_column(softplus, &mut physical, 5)?;
        apply_transform_column(logistic, &mut physical, 6)?;

        apply_transform_column(softplus, &mut physical, 7)?;

        Ok(())
    }

    fn convert_internal_to_physical_tracking_ll_warp(
        &self,
        log_jacobian: &mut f64,
        internal: ArrayView1<f64>,
        physical: &mut ndarray::prelude::ArrayViewMut1<f64>,
    ) -> anyhow::Result<()> {
        physical.assign(&internal);

        logistic_log_jacobian_inplace(&mut physical[0], log_jacobian)?;
        softplus_log_jacobian_inplace(&mut physical[1], log_jacobian)?;
        softplus_log_jacobian_inplace(&mut physical[3], log_jacobian)?;
        softplus_log_jacobian_inplace(&mut physical[4], log_jacobian)?;
        softplus_log_jacobian_inplace(&mut physical[5], log_jacobian)?;
        logistic_log_jacobian_inplace(&mut physical[6], log_jacobian)?;

        softplus_log_jacobian_inplace(&mut physical[7], log_jacobian)?;

        Ok(())
    }

    fn log_likelihood(&self, parameters: ArrayView1<f64>) -> anyhow::Result<f64> {
        let theta = parameters[0]; // fractional weight of exponential 
        let alpha = parameters[1]; // exponential scale height
        let mu_rc = parameters[2]; // mean of RC gaussian  component 
        // Gaussian standard deviations
        let sigma_rc = parameters[3];

        // RC -- RGBB parameters
        let d_mu = parameters[4]; // Delta mag [RGBB - RC]
        let f_sigma = parameters[5]; // Ratio of RGBB / RC sigma
        let w_rc = parameters[6]; // Weight of RC [RC_w + RGBB_w = 1] 

        // Hyper-parameter controlling usage of second gaussian
        let tau = parameters[7];

        let log_exp_int = -alpha.ln() + log_diff_exp(alpha * self.ks_max, alpha * self.ks_min)?;

        let ln_theta = theta.ln();
        let ln_1_sub_theta = (1.0 - theta).ln();

        let ln_w_rc = w_rc.ln();
        let ln_1_sub_w_rc = (1.0 - w_rc).ln();

        let mu_rgbb = mu_rc + d_mu;
        let sigma_rgbb = f_sigma * sigma_rc;

        let mut ll = 0.0;

        // prior on tau
        ln_prior_exponential(tau, 1.0, &mut ll);

        // priors on the RGBB components
        // this uses a hierarchical prior to push this normal dist prior towards a dispersion of 0
        ln_prior_normal(1.0 - w_rc, 0.0, tau, &mut ll);

        ln_prior_normal(d_mu, 0.8, 0.1, &mut ll);
        ln_prior_normal(f_sigma, 1.0, 0.2, &mut ll);

        for (ks, w) in self.ks.iter().zip(self.w_rc.iter()) {
            let ll_g = log_sum_exp(
                ln_w_rc + log_normalised_gaussian(*ks, mu_rc, sigma_rc),
                ln_1_sub_w_rc + log_normalised_gaussian(*ks, mu_rgbb, sigma_rgbb),
            )
            .unwrap_or(f64::NEG_INFINITY);

            let ll_mix = log_sum_exp(
                ln_theta + ll_g,
                ln_1_sub_theta + (alpha * *ks - log_exp_int),
            )
            .unwrap_or(f64::NEG_INFINITY);

            ll += w * ll_mix
        }

        Ok(ll)
    }
}
