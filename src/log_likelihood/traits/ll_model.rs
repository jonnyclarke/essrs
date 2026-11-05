use ndarray::ArrayView1;

use crate::log_likelihood::ParameterTransformer;

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
///     return `f64::NEG_INFINITY`
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
/// use essrs::log_likelihood::{LogLikelihoodModel, ParameterTransformer, Parametrised};
///
/// struct NormalOneDim {
///     x: f64
/// }
///
/// impl Parametrised for NormalOneDim {
///     fn get_n_parameters(&self) -> usize {2}
/// }
///
/// impl ParameterTransformer for NormalOneDim {}
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
///     log_likelihood::{LogLikelihoodModel, Parametrised, ParameterTransformer}
/// };
///
/// struct NormalOneDim {
///     x: f64
/// }
///
/// impl Parametrised for NormalOneDim {
///     fn get_n_parameters(&self) -> usize {2}
/// }
///
/// impl ParameterTransformer for NormalOneDim {
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
/// }
///
/// impl LogLikelihoodModel for NormalOneDim {
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
pub trait LogLikelihoodModel: ParameterTransformer + Send + Sync {
    /// Returns the log-likelihood for a given set of parameters.
    ///
    /// Constraints:
    /// - system will return an error if log-likelihood returns NaN or +INF
    /// - system will interpret -INF as a hard boundary in phase space
    ///
    /// # Errors
    /// * failure of log-likelihood implementation
    fn log_likelihood(&self, parameters: ArrayView1<f64>) -> anyhow::Result<f64>;
}
