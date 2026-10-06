//! Log-likelihood models.
//! Includes a library of common non-trivial log-likelihood functions for easy use.
//!
//! A note on parameter-spaces.
//! The MCMC sampler utilises two separate parameter spaces.
//! The first is the user-facing space. This is the parameter space in which all initial conditions are given and in which all outputs will be given. For example in the case of a 2 dimensional gaussian the parameters are, mu1, mu2, s1, s2, and p (means in dimension 1 and 2, standard deviations in dimension 1 and 2, and correlation coefficient).
//! These are not the natural coordinates in which to sample since there are hard limitations on parameter values:
//! s1 > 0, s2 > 0, -1 <= p <= 1
//! It therefore makes sense to sample in a secondary phase space and transform back to user-facing scheme for log-likelihood evaluation and for output. A clear example of this is applying the sigmoid function to a free parameter space to mathematically bound it to -1 and 1. As long as the log likelihood is modified to account for this distortion the sampling process is far more robust given we do not have to implement arbitrary walls where the log-likelihood goes to negative infinity.
//! We therefore define two additional routines here, the default for which are to do nothing at all.
//! The first is to apply the inverse transformation to the initial conditions.
//! The second is to apply the transformation

pub mod normal_1d;
pub mod normal_2d;

use anyhow;
use ndarray::{Array1, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LogLikelihoodError {
    #[error("log likelihood determined to be NaN for parameters: {params}")]
    LogLikelihoodIsNan { params: Array1<f64> },

    #[error("log likelihood determined to be infinite for parameters: {params}")]
    LogLikelihoodIsInfinite { params: Array1<f64> },

    #[error("{context_string}:\n{source}")]
    UserDefLogLikelihoodFnError {
        context_string: &'static str,

        #[source]
        source: anyhow::Error,
    },
}

/// Trait for valid log-likelihood models.
pub trait LogLikelihoodModel {
    /// Transform initial conditions into the internal phase space
    fn columnar_transform_physical_to_internal(
        &self,
        physical: ArrayView2<f64>,
        mut internal: ArrayViewMut2<f64>,
    ) -> anyhow::Result<()> {
        internal.assign(&physical);

        Ok(())
    }

    fn columnar_transform_internal_to_physical(
        &self,
        internal: ArrayView2<f64>,
        mut physical: ArrayViewMut2<f64>,
    ) -> anyhow::Result<()> {
        physical.assign(&internal);

        Ok(())
    }

    /// Transform parameter vector into physical space from internal space.
    /// This function also returns the value of the log-likelihood modification due to the phase space distortion
    fn convert_internal_to_physical_tracking_ll_warp(
        &self,
        log_jacobian: &mut f64,
        internal: ArrayView1<f64>,
        physical: &mut ArrayViewMut1<f64>,
    ) -> anyhow::Result<()> {
        *log_jacobian = 0.0;
        physical.assign(&internal);

        Ok(())
    }

    /// Returns the log-likelihood for a given set of parameters.
    fn log_likelihood(&self, parameters: ArrayView1<f64>) -> anyhow::Result<f64>;
}

pub trait WrappedLogLikelihoodModel: LogLikelihoodModel {
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

        if ll_parameters.is_infinite() {
            return Err(LogLikelihoodError::LogLikelihoodIsInfinite {
                params: physical.to_owned(),
            });
        }

        Ok(log_jacobian + ll_parameters)
    }
}

impl<T: LogLikelihoodModel + ?Sized> WrappedLogLikelihoodModel for T {}
