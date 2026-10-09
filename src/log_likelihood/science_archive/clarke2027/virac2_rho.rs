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
        {apply_transform_column, apply_inplace_multi_column_transform},
        logistic::{logistic_log_jacobian_inplace, logistic_inverse, logistic},
        softplus::{softplus_log_jacobian_inplace, softplus_inverse, softplus},
        ordseq::{ordinal_sequencer_inverse, ordinal_sequencer}
    },
};


pub struct Clarke2027LogLiViracV2Density {
    ks: Array1<f64>,
    // n: usize,

    ks_min: f64,
    ks_max: f64,
}

impl Clarke2027LogLiViracV2Density {
    pub fn new(
        ks: Array1<f64>,
        ks_min: f64,
        ks_max: f64
    ) -> Self {
        // let n = ks.len();
        Self {
            ks,
            // n,
            ks_min,
            ks_max
        }
    }
}

/// Parameters of this model:
/// - 0: theta   :: weight between the exponential and gaussian mixture
/// - 1: alpha   :: exponential scale height
/// - 2: pi      :: weight of 1^th Gaussian
/// - 3: mu1     :: mean of Gaussian component 1 [Red Clump]
/// - 4: mu2     :: mean of gaussian component 2 [Red Giant Branch Bump]
/// - 5: sigma1  :: standard deviation of gaussian 1
/// - 6: sigma2  :: standard deviation of gaussian 2
/// - 7: d_mu    :: Delta central magnitude [RGBB - RC]
/// - 8: f_sigma :: Ratio of RGBB to RC dispersion [constrained >0]
/// - 9: w_rc    :: Weight of RC [RC_w + RGBB_w = 1] 
impl LogLikelihoodModel for Clarke2027LogLiViracV2Density {

    fn columnar_transform_physical_to_internal(
        &self,
        physical: ndarray::prelude::ArrayView2<f64>,
        mut internal: ndarray::prelude::ArrayViewMut2<f64>,
    ) -> anyhow::Result<()>
    {
        internal.assign(&physical);

        apply_transform_column(logistic_inverse, &mut internal, 0)?;
        
        apply_transform_column(softplus_inverse, &mut internal, 1)?;

        apply_transform_column(logistic_inverse, &mut internal, 2)?;

        // we do not modify column 3, column 4 is ordered based on column 3
        apply_inplace_multi_column_transform(ordinal_sequencer_inverse, &mut internal, 3, 4)?;
        apply_transform_column(softplus_inverse, &mut internal, 4)?;

        // apply softplus inversion to dispersions
        apply_transform_column(softplus_inverse, &mut internal, 5)?;
        apply_transform_column(softplus_inverse, &mut internal, 6)?;

        apply_transform_column(softplus_inverse, &mut internal, 7)?;
        apply_transform_column(softplus_inverse, &mut internal, 8)?;
        apply_transform_column(logistic_inverse, &mut internal, 9)?;
        

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

        apply_transform_column(logistic, &mut physical, 2)?;

        // we do not modify column 3, column 4 is ordered based on column 3
        apply_transform_column(softplus, &mut physical, 4)?;
        apply_inplace_multi_column_transform(ordinal_sequencer, &mut physical, 3, 4)?;

        // apply softplus inversion to dispersions
        apply_transform_column(softplus, &mut physical, 5)?;
        apply_transform_column(softplus, &mut physical, 6)?;

        apply_transform_column(softplus, &mut physical, 7)?;
        apply_transform_column(softplus, &mut physical, 8)?;
        apply_transform_column(logistic, &mut physical, 9)?;

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

        logistic_log_jacobian_inplace(&mut physical[2], log_jacobian)?;

        softplus_log_jacobian_inplace(&mut physical[4], log_jacobian)?;
        ordinal_sequencer(physical[3], &mut physical[4])?;

        softplus_log_jacobian_inplace(&mut physical[5], log_jacobian)?;
        softplus_log_jacobian_inplace(&mut physical[6], log_jacobian)?;

        softplus_log_jacobian_inplace(&mut physical[7], log_jacobian)?;
        softplus_log_jacobian_inplace(&mut physical[8], log_jacobian)?;
        logistic_log_jacobian_inplace(&mut physical[9], log_jacobian)?;

        Ok(())
    }

    fn log_likelihood(
            &self,
            parameters: ArrayView1<f64>,
        ) -> anyhow::Result<f64> {

        let theta = parameters[0]; // fractional weight of exponential 
        let alpha = parameters[1]; // exponential scale height
        let pi = parameters[2]; // weight of gaussian 0
        // Gaussian means
        let mu0 = parameters[3]; // gaussian mean component 1
        let mu1 = parameters[4]; // gaussian mean component 2
        // Gaussian standard deviations
        let sigma0 = parameters[5];
        let sigma1 = parameters[6];

        // RC -- RGBB parameters
        let d_mu = parameters[7];     // Delta mag [RGBB - RC]
        let f_sigma = parameters[8];  // Ratio of RGBB / RC sigma
        let w_rc = parameters[9];     // Weight of RC [RC_w + RGBB_w = 1] 

        let log_exp_int = -alpha.ln() + log_diff_exp(
            alpha * self.ks_max,
            alpha * self.ks_min
        )?;

        let mut ll = 0.0;

        for ks in self.ks.iter() {
            let ll_g = log_sum_exp(
                pi.ln() + log_sum_exp(
                    w_rc.ln() + log_normalised_gaussian(*ks, mu0, sigma0),
                    (1.0 - w_rc).ln() + log_normalised_gaussian(*ks, mu0 + d_mu, sigma0 * f_sigma)
                )?,
                (1.0 - pi).ln() + log_sum_exp(
                    w_rc.ln() + log_normalised_gaussian(*ks, mu1, sigma1),
                    (1.0 - w_rc).ln() + log_normalised_gaussian(*ks, mu1 + d_mu, sigma1 * f_sigma)
                )?
            )?;

            let ll_mix = log_sum_exp(
                theta.ln() + ll_g,
                (1.0 - theta).ln() + (alpha * *ks - log_exp_int)
            )?;

            ll += ll_mix
        }

        // PARALLEL VERSION ...

        // get rid of this allocation each call
        // let ll_exp = alpha * &self.ks - log_exp_int;

        // ll += (0..self.n).into_par_iter().map(|i| {

        //     let ll_gmix = log_sum_exp(
        //         pi.ln() + log_sum_exp(
        //             w_rc.ln() + log_normalised_gaussian(self.ks[i], mu0, sigma0),
        //             (1.0 - w_rc).ln() + log_normalised_gaussian(self.ks[i], mu0 + d_mu, sigma0 * f_sigma)
        //         ).unwrap(),
        //         (1.0 - pi).ln() + log_sum_exp(
        //             w_rc.ln() + log_normalised_gaussian(self.ks[i], mu1, sigma1),
        //             (1.0 - w_rc).ln() + log_normalised_gaussian(self.ks[i], mu1 + d_mu, sigma1 * f_sigma)
        //         ).unwrap()
        //     ).unwrap();

        //     log_sum_exp(
        //         theta.ln() + ll_gmix,
        //         (1.0 - theta).ln() + ll_exp[i]
        //     ).unwrap()
        // }).sum::<f64>();

        Ok(ll)
    }
}

