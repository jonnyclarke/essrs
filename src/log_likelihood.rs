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
use ndarray::{Array1, Array2, ArrayView1};
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
        parameters: &Array2<f64>,
    ) -> anyhow::Result<Array2<f64>> {
        Ok(parameters.to_owned())
    }

    fn columnar_transform_internal_to_physical(
        &self,
        parameters: &Array2<f64>,
    ) -> anyhow::Result<Array2<f64>> {
        Ok(parameters.to_owned())
    }

    /// Transform parameter vector into physical space from internal space.
    /// This function also returns the value of the log-likelihood modification due to the phase space distortion
    fn convert_internal_to_physical_tracking_ll_warp(
        &self,
        parameters: ArrayView1<f64>,
    ) -> anyhow::Result<(f64, Array1<f64>)> {
        Ok((0.0, parameters.to_owned()))
    }

    /// Returns the log-likelihood for a given set of parameters.
    fn log_likelihood(&self, parameters: &Array1<f64>) -> anyhow::Result<f64>;
}

pub trait WrappedLogLikelihoodModel: LogLikelihoodModel {
    fn wrapped_columnar_transform_physical_to_internal(
        &self,
        parameters: &Array2<f64>,
    ) -> Result<Array2<f64>, LogLikelihoodError> {
        let x = self.columnar_transform_physical_to_internal(parameters)
            .map_err(
                |source| LogLikelihoodError::UserDefLogLikelihoodFnError {
                    context_string: "Parameter conversion, columnwise, from physical to internal scale has failed",
                    source,
                }
            )?;

        Ok(x)
    }

    fn wrapped_columnar_transform_internal_to_physical(
        &self,
        parameters: &Array2<f64>,
    ) -> Result<Array2<f64>, LogLikelihoodError> {
        let x = self.columnar_transform_internal_to_physical(parameters)
            .map_err(
                |source| LogLikelihoodError::UserDefLogLikelihoodFnError {
                    context_string: "Parameter conversion, columnwise, from internal to physical scale has failed",
                    source,
                }
            )?;

        Ok(x)
    }

    fn wrapped_convert_internal_to_physical_tracking_ll_warp(
        &self,
        parameters: ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), LogLikelihoodError> {
        let x = self.convert_internal_to_physical_tracking_ll_warp(parameters)
            .map_err(
                |source| LogLikelihoodError::UserDefLogLikelihoodFnError {
                    context_string: "parameter conversion, tracking log-jacobian contribution, from internal to physical scale has failed",
                    source,
                }
            )?;

        Ok(x)
    }

    /// DO NOT RE-IMPLEMENT! -- extract to a different trait that cannot be re-implemented.
    /// this is the only time the users log-likelihood function is called and we wrap it in order to
    fn wrapped_log_likelihood(
        &self,
        parameters: ArrayView1<f64>,
    ) -> Result<f64, LogLikelihoodError> {
        let (ll_phase_transform, modified_parameters) =
            self.wrapped_convert_internal_to_physical_tracking_ll_warp(parameters)?;

        let ll_parameters = self
            .log_likelihood(&modified_parameters)
            .map_err(|source| LogLikelihoodError::UserDefLogLikelihoodFnError {
                context_string: "user-defined log-likelihood function has failed",
                source,
            })?;

        if ll_parameters.is_nan() {
            return Err(LogLikelihoodError::LogLikelihoodIsNan {
                params: modified_parameters,
            });
        }

        if ll_parameters.is_infinite() {
            return Err(LogLikelihoodError::LogLikelihoodIsInfinite {
                params: modified_parameters,
            });
        }

        Ok(ll_phase_transform + ll_parameters)
    }
}

impl<T: LogLikelihoodModel + ?Sized> WrappedLogLikelihoodModel for T {}
