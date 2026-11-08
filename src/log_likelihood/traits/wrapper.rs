use anyhow::{Context, ensure};
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

use crate::log_likelihood::LogLikelihoodModel;

pub(crate) trait WrappedLogLikelihoodModel: LogLikelihoodModel {
    fn wrapped_columnar_transform_physical_to_internal(
        &self,
        physical: ArrayView2<f64>,
        internal: ArrayViewMut2<f64>,
    ) -> anyhow::Result<()> {
        self.columnar_transform_physical_to_internal(physical, internal)
            .context(
                "parameter conversion, columnwise, from physical to internal scale has failed",
            )?;

        Ok(())
    }

    fn wrapped_columnar_transform_internal_to_physical(
        &self,
        internal: ArrayView2<f64>,
        physical: ArrayViewMut2<f64>,
    ) -> anyhow::Result<()> {
        self.columnar_transform_internal_to_physical(internal, physical)
            .context(
                "parameter conversion, columnwise, from internal to physical scale has failed",
            )?;

        Ok(())
    }

    fn wrapped_convert_internal_to_physical_tracking_ll_warp(
        &self,
        log_jacobian: &mut f64,
        internal: ArrayView1<f64>,
        physical: &mut ArrayViewMut1<f64>,
    ) -> anyhow::Result<()> {
        self.convert_internal_to_physical_tracking_ll_warp(log_jacobian, internal, physical)
            .context("parameter conversion, tracking log-jacobian contribution, from internal to physical scale has failed")?;

        Ok(())
    }

    /// DO NOT RE-IMPLEMENT! -- extract to a different trait that cannot be re-implemented.
    /// this is the only time the users log-likelihood function is called and we wrap it in order to
    fn wrapped_log_likelihood(
        &self,
        internal: ArrayView1<f64>,
        physical: &mut ArrayViewMut1<f64>,
    ) -> anyhow::Result<f64> {
        let mut log_jacobian: f64 = 0.0;

        self.wrapped_convert_internal_to_physical_tracking_ll_warp(
            &mut log_jacobian,
            internal,
            physical,
        )
        .context("user-defined parameter transformation, tracking log-jacobian warp, has failed")?;

        let ll_parameters = self
            .log_likelihood(physical.view())
            .context("user defined log-likelihood function has failed")?;

        ensure!(
            !ll_parameters.is_nan(),
            "user defined log-likelihood returns NaN, parameters:\n\t[{:#?}]",
            physical.to_owned()
        );

        ensure!(
            ll_parameters.ne(&f64::INFINITY),
            "user defined log-likelihood returns +INF, parameters:\n\t[{:#?}]",
            physical.to_owned()
        );

        Ok(log_jacobian + ll_parameters)
    }
}

impl<T: LogLikelihoodModel + ?Sized> WrappedLogLikelihoodModel for T {}
