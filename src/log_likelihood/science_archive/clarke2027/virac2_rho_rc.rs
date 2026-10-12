/// Log likelihood function for fitting VIRAC-2 density profiles with
/// an exponential and gaussian mixture

use ndarray::{Array1, ArrayView1};
use crate::log_likelihood::LogLikelihoodModel;
use crate::functions::{
    numerical::{
        log_diff_exp,
        log_sum_exp
    },
    likelihood::log_normalised_gaussian,
    transforms::{
        apply_transform_column,
        logistic::{logistic_log_jacobian_inplace, logistic_inverse, logistic},
        softplus::{softplus_log_jacobian_inplace, softplus_inverse, softplus},
    },
};


pub struct Clarke2027LogLiViracV2Rc {
    ks: Array1<f64>,
    w_rc: Array1<f64>,
    ks_min: f64,
    ks_max: f64,
}

impl Clarke2027LogLiViracV2Rc {
    pub fn new(
        ks: Array1<f64>,
        w_rc: Array1<f64>,
        ks_min: f64,
        ks_max: f64
    ) -> Self {
        Self {
            ks,
            w_rc,
            ks_min,
            ks_max
        }
    }
}

/// Parameters of this model:
/// - 0: theta   :: weight between the exponential and gaussian mixture
/// - 1: alpha   :: exponential scale height
/// - 2: mu      :: mean of RC Gaussian component 
/// - 3: sigma   :: standard deviation of RC gaussian
impl LogLikelihoodModel for Clarke2027LogLiViracV2Rc {

    fn columnar_transform_physical_to_internal(
        &self,
        physical: ndarray::prelude::ArrayView2<f64>,
        mut internal: ndarray::prelude::ArrayViewMut2<f64>,
    ) -> anyhow::Result<()>
    {
        internal.assign(&physical);

        apply_transform_column(logistic_inverse, &mut internal, 0)?;
        apply_transform_column(softplus_inverse, &mut internal, 1)?;
        apply_transform_column(softplus_inverse, &mut internal, 3)?;

        Ok(())
    }

    fn columnar_transform_internal_to_physical(
        &self,
        internal: ndarray::prelude::ArrayView2<f64>,
        mut physical: ndarray::prelude::ArrayViewMut2<f64>,
    ) -> anyhow::Result<()>
    {
        physical.assign(&internal);

        apply_transform_column(logistic, &mut physical, 0)?;
        apply_transform_column(softplus, &mut physical, 1)?;
        apply_transform_column(softplus, &mut physical, 3)?;


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

        logistic_log_jacobian_inplace(&mut physical[0], log_jacobian)?;
        softplus_log_jacobian_inplace(&mut physical[1], log_jacobian)?;
        softplus_log_jacobian_inplace(&mut physical[3], log_jacobian)?;

        Ok(())
    }

    fn log_likelihood(
            &self,
            parameters: ArrayView1<f64>,
        ) -> anyhow::Result<f64> {

        let theta = parameters[0]; // fractional weight of exponential 
        let alpha = parameters[1]; // exponential scale height
        let mu_rc = parameters[2]; // mean of RC gaussian  component 
        // Gaussian standard deviations
        let sigma_rc = parameters[3];

        let log_exp_int = -alpha.ln() + log_diff_exp(
            alpha * self.ks_max,
            alpha * self.ks_min
        )?;

        let ln_theta = theta.ln();
        let ln_1_sub_theta = (1.0 - theta).ln();

        let mut ll = 0.0;

        for (ks, w) in self.ks.iter().zip(self.w_rc.iter()) {

            let ll_mix = log_sum_exp(
                ln_theta + log_normalised_gaussian(*ks, mu_rc, sigma_rc),
                ln_1_sub_theta + (alpha * *ks - log_exp_int)
            ).unwrap_or(f64::NEG_INFINITY);

            ll += w * ll_mix
        }

        Ok(ll)
    }
}

