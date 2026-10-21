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

mod wrapper;
pub(crate) use wrapper::WrappedLogLikelihoodModel;

mod helpers;
pub use helpers::helper_generate_random_gaussian_points;

pub mod normal_1d;
pub mod normal_2d;

use anyhow;
use ndarray::{Array1, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};
use thiserror::Error;

/// Possible errors of the User Defined Log-Likelihood function
#[derive(Debug, Error)]
pub enum LogLikelihoodError {
    /// Log likelihood is NaN
    #[error("log likelihood determined to be NaN for parameters: {params}")]
    LogLikelihoodIsNan { params: Array1<f64> },

    /// Log Likelihood is + infinite
    /// NOTE: a -inf log-likelihood is tolerated by the system and treated as an impossible region
    #[error("log likelihood determined to be infinite for parameters: {params}")]
    LogLikelihoodIsInfinite { params: Array1<f64> },

    /// Error in the Log-likelihood function itself
    /// this maps any error that occurs in the UserDefined function back into the crate error system
    #[error("{context_string}:\n{source}")]
    UserDefLogLikelihoodFnError {
        context_string: &'static str,

        #[source]
        source: anyhow::Error,
    },
}

/// Trait for valid log-likelihood models.
/// NOTE: Log-likelihood structure must not have a mutable internal state in order to use the parallelisation
/// capabilities of the defined moves.
///
/// The primary function is the user defined log-likelihood function which must be implemented.
///
/// There are a further three optional phase-space manipulation functions which can be used at users discretion:
/// * convert physical initial conditions to internal unconstrained respresentation
/// * convert back from internal to physical phase space
/// * convert a parameter array from internal to physical and track the adjustment to the log-likelihood to account for the different phase space volumes.
///
/// # Notes
///
/// Consider the following trivial problem.
/// The user is fitting a 1d normal distribution to a set of points.
/// There are two free parameters: mu and sigma.
/// mu    : unconstrained
/// sigma : constrained to >0
///
/// The simplest way is to sample both values in physical space and set a hard prior of the type:
/// '''
/// if sigma <= 0 {
///     return f64::NEG_INFINITY
/// }
/// '''
/// However, if sigma is very small, this prior can impede convergence.
///
/// A more elegant approach is to sample in an internal space.
/// Sample a hyper-parameter, phi, in continuous space and apply the softplus transform.
/// Thus: -inf < phi < + inf  ==> 0 < sigma
/// In this scenario phi is a far more natural phase space to sample as it automatically handles the case where sigma is very small.
/// A caveat of this approach is that it requires more book keeping to handle the internal -> physical transition as well as adjusting the log-likelihood to account for the transition.
///
/// # Examples
///
/// ```
/// // Using hard boundary prior
/// use ndarray::ArrayView1;
///
/// use essrs::functions::likelihood::log_normalised_gaussian;
/// use essrs::log_likelihood::LogLikelihoodModel;
///
/// struct NormalOneDim {
///     x: f64
/// }
///
/// impl LogLikelihoodModel for NormalOneDim {
///     fn log_likelihood(&self, parameters: ArrayView1<f64>) -> anyhow::Result<f64> {
///         let mu = parameters[0];
///         let sigma = parameters[1];
///         if sigma <= 0.0 {
///             return Ok(f64::NEG_INFINITY);
///         }
///
///         let ll = log_normalised_gaussian(self.x, mu, sigma);
///
///         Ok(ll)
///     }
/// }
/// ```
///
/// ```
/// // Using internal phase space representation
/// use ndarray::{ArrayView1, ArrayViewMut1, ArrayView2, ArrayViewMut2};
///
/// use essrs::{
///     functions::{
///         likelihood::log_normalised_gaussian,
///         transforms::{
///             apply_transform_column,
///             softplus::{softplus, softplus_inverse, softplus_log_jacobian_inplace},
///         },
///     },
///     log_likelihood::LogLikelihoodModel,
/// };
///
/// struct NormalOneDim {
///     x: f64
/// }
///
/// impl LogLikelihoodModel for NormalOneDim {
///     fn columnar_transform_physical_to_internal(
///         &self,
///         physical: ArrayView2<f64>,
///         mut internal: ArrayViewMut2<f64>,
///     ) -> anyhow::Result<()> {
///         internal.assign(&physical);
///         apply_transform_column(softplus_inverse, &mut internal, 1)?;
///
///         Ok(())
///     }
///
///     fn columnar_transform_internal_to_physical(
///         &self,
///         internal: ArrayView2<f64>,
///         mut physical: ArrayViewMut2<f64>,
///     ) -> anyhow::Result<()> {
///         physical.assign(&internal);
///         apply_transform_column(softplus, &mut physical, 1)?;
///
///         Ok(())
///     }
///
///     fn convert_internal_to_physical_tracking_ll_warp(
///         &self,
///         log_jacobian: &mut f64,
///         internal: ArrayView1<f64>,
///         physical: &mut ArrayViewMut1<f64>,
///     ) -> anyhow::Result<()> {
///         physical.assign(&internal);
///
///         softplus_log_jacobian_inplace(&mut physical[1], log_jacobian)?;
///
///         Ok(())
///     }
///
///     fn log_likelihood(&self, parameters: ArrayView1<f64>) -> anyhow::Result<f64> {
///         let mu = parameters[0];
///         let sigma = parameters[1];
///
///         let ll = log_normalised_gaussian(self.x, mu, sigma);
///
///         Ok(ll)
///     }
/// }
/// ```
pub trait LogLikelihoodModel: Send + Sync {
    /// Transform physical phase space to internal
    /// Used when ingesting initial conditions provided by the user
    ///
    /// The user is responsible for handling parameter ordering when implementing this function.
    /// If no transform is needed use the default implementation which handles book keeping of internal data structures.
    ///
    /// # Arguments
    /// * physical: initial conditions provided with shape [N_walkers, N_parameters]
    /// * internal: mutable array onto which the transformed parameters will be assigned
    fn columnar_transform_physical_to_internal(
        &self,
        physical: ArrayView2<f64>,
        mut internal: ArrayViewMut2<f64>,
    ) -> anyhow::Result<()> {
        internal.assign(&physical);

        Ok(())
    }

    /// Transform internal phase space to physical
    /// Used when storing an iteration into the chain storage
    ///
    /// The user is responsible for handling parameter ordering when implementing this function.
    /// If no transform is needed use the default implementation which handles book keeping of internal data structures.
    ///
    /// # Arguments
    /// * internal: internal walker state of shape [N_walkers, N_parameters]
    /// * physical: mutable array onto which the transformed parameters will be assigned
    fn columnar_transform_internal_to_physical(
        &self,
        internal: ArrayView2<f64>,
        mut physical: ArrayViewMut2<f64>,
    ) -> anyhow::Result<()> {
        physical.assign(&internal);

        Ok(())
    }

    /// Transform parameter vector from internal to physical phase space and compute log-likelihood term from log-jacobian
    /// This function is used within the MCMC process to convert proposals back to physical space for log-likelihood function
    ///
    /// # Arguments
    /// * log_jacobian: mutable reference which should be modified according to log-likelihood warp
    /// * internal: array view of proposal parameters in internal phase space
    /// * physical: mutable array reference for assignment of proposal parameters in physical phase space
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
    ///
    /// Constraints:
    /// - system will return an error if log-likelihood returns NaN or +INF
    /// - system will interpret -INF as a hard boundary in phase space
    fn log_likelihood(&self, parameters: ArrayView1<f64>) -> anyhow::Result<f64>;
}
