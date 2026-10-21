use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

use crate::log_likelihood::{LogLikelihoodError, LogLikelihoodModel};

pub(crate) trait WrappedLogLikelihoodModel: LogLikelihoodModel {
    fn wrapped_columnar_transform_physical_to_internal(
        &self,
        physical: ArrayView2<f64>,
        internal: ArrayViewMut2<f64>,
    ) -> Result<(), LogLikelihoodError> {
        self.columnar_transform_physical_to_internal(physical, internal)
            .map_err(
                |source| LogLikelihoodError::UserDefLogLikelihoodFnError {
                    context_string: "Parameter conversion, columnwise, from physical to internal scale has failed",
                    source,
                }
            )?;

        Ok(())
    }

    fn wrapped_columnar_transform_internal_to_physical(
        &self,
        internal: ArrayView2<f64>,
        physical: ArrayViewMut2<f64>,
    ) -> Result<(), LogLikelihoodError> {
        self.columnar_transform_internal_to_physical(internal, physical)
            .map_err(
                |source| LogLikelihoodError::UserDefLogLikelihoodFnError {
                    context_string: "Parameter conversion, columnwise, from internal to physical scale has failed",
                    source,
                }
            )?;

        Ok(())
    }

    fn wrapped_convert_internal_to_physical_tracking_ll_warp(
        &self,
        log_jacobian: &mut f64,
        internal: ArrayView1<f64>,
        physical: &mut ArrayViewMut1<f64>,
    ) -> Result<(), LogLikelihoodError> {
        self.convert_internal_to_physical_tracking_ll_warp(log_jacobian, internal, physical)
            .map_err(
                |source| LogLikelihoodError::UserDefLogLikelihoodFnError {
                    context_string: "parameter conversion, tracking log-jacobian contribution, from internal to physical scale has failed",
                    source,
                }
            )?;

        Ok(())
    }

    /// DO NOT RE-IMPLEMENT! -- extract to a different trait that cannot be re-implemented.
    /// this is the only time the users log-likelihood function is called and we wrap it in order to
    fn wrapped_log_likelihood(
        &self,
        internal: ArrayView1<f64>,
        physical: &mut ArrayViewMut1<f64>,
    ) -> Result<f64, LogLikelihoodError> {
        let mut log_jacobian: f64 = 0.0;

        self.wrapped_convert_internal_to_physical_tracking_ll_warp(
            &mut log_jacobian,
            internal,
            physical,
        )?;

        let ll_parameters = self.log_likelihood(physical.view()).map_err(|source| {
            LogLikelihoodError::UserDefLogLikelihoodFnError {
                context_string: "user-defined log-likelihood function has failed",
                source,
            }
        })?;

        if ll_parameters.is_nan() {
            return Err(LogLikelihoodError::LogLikelihoodIsNan {
                params: physical.to_owned(),
            });
        }

        if ll_parameters.is_infinite() && ll_parameters.is_sign_positive() {
            return Err(LogLikelihoodError::LogLikelihoodIsInfinite {
                params: physical.to_owned(),
            });
        }

        Ok(log_jacobian + ll_parameters)
    }
}

impl<T: LogLikelihoodModel + ?Sized> WrappedLogLikelihoodModel for T {}
