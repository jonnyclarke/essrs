use anyhow;
use ndarray::{ArrayView1, ArrayViewMut1};
use statrs::distribution::{Beta, Continuous};

use crate::{
    functions::transforms::{
        apply_transform_column,
        logistic::{logistic, logistic_inverse, logistic_log_jacobian_inplace},
    },
    log_likelihood::{LogLikelihoodModel, ProbDataGivenThetaModel},
};

const N_PARAMETERS_BETA_DISTRIBUTUON: usize = 1;

#[cfg_attr(all(doc, feature = "doc-math"), katexit::katexit)]
/// Likelihood construction of the beta distribution toy problem.
///
/// The beta distribution is defined as:
///
/// $$
/// f\left(x|\alpha, \beta\right) = \frac{x^{\alpha - 1}\left(1-x\right)^{\beta-1}}{\mathcal{B}\left(\alpha, \beta\right)}
/// $$
///
/// where
///
/// $$
/// \mathcal{B}\left(\alpha, \beta\right) = \frac{\Gamma\left(\alpha\right) \Gamma\left(\beta\right)}{\Gamma\left(\alpha + \beta\right)}
/// $$
///
/// The default parameter values are $$ \alpha=0.5, \beta=0.5 $$ which produces a distinctive U-shaped distribution with asymptotes to +inf at 0 and 1.
pub struct AnalyticBetaDistToyLogL {
    _alpha: f64,
    _beta: f64,
    f: Beta,
}

impl AnalyticBetaDistToyLogL {
    /// Constructor
    ///
    /// # Errors
    /// -
    pub fn new(alpha: f64, beta: f64) -> anyhow::Result<Self> {
        let f = Beta::new(alpha, beta)?;
        Ok(Self {
            _alpha: alpha,
            _beta: beta,
            f,
        })
    }
}

impl Default for AnalyticBetaDistToyLogL {
    fn default() -> Self {
        Self::new(0.5, 0.5).expect("default values caused initialisation failure")
    }
}

impl LogLikelihoodModel for AnalyticBetaDistToyLogL {
    fn get_n_parameters(&self) -> usize {
        N_PARAMETERS_BETA_DISTRIBUTUON
    }

    fn columnar_transform_physical_to_internal(
        &self,
        physical: ndarray::prelude::ArrayView2<f64>,
        mut internal: ndarray::prelude::ArrayViewMut2<f64>,
    ) -> anyhow::Result<()> {
        internal.assign(&physical);
        apply_transform_column(logistic_inverse, &mut internal, 0)?;

        Ok(())
    }

    fn columnar_transform_internal_to_physical(
        &self,
        internal: ndarray::prelude::ArrayView2<f64>,
        mut physical: ndarray::prelude::ArrayViewMut2<f64>,
    ) -> anyhow::Result<()> {
        physical.assign(&internal);
        apply_transform_column(logistic, &mut physical, 0)?;

        Ok(())
    }

    fn convert_internal_to_physical_tracking_ll_warp(
        &self,
        log_jacobian: &mut f64,
        internal: ArrayView1<f64>,
        physical: &mut ArrayViewMut1<f64>,
    ) -> anyhow::Result<()> {
        physical.assign(&internal);

        logistic_log_jacobian_inplace(&mut physical[0], log_jacobian)?;

        Ok(())
    }

    fn log_likelihood(&self, parameters: ArrayView1<f64>) -> anyhow::Result<f64> {
        Ok(self.f.ln_pdf(parameters[0]))
    }
}

impl ProbDataGivenThetaModel for AnalyticBetaDistToyLogL {}

#[cfg(test)]
mod tests {

    use ndarray::array;

    use super::*;

    #[test]
    fn test_initialisation() -> anyhow::Result<()> {
        let ll = AnalyticBetaDistToyLogL::default();
        let parameters = array![0.5];
        let _ = ll.log_likelihood(parameters.view());

        Ok(())
    }

    #[test]
    fn test_get_n_parameters() -> anyhow::Result<()> {
        let ll = AnalyticBetaDistToyLogL::default();

        assert_eq!(ll.get_n_parameters(), N_PARAMETERS_BETA_DISTRIBUTUON);

        Ok(())
    }
}
