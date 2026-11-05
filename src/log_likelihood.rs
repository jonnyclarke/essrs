//! A library of pre-implemented log-likelihood functions for general use.
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
pub mod f_scientific;
pub mod f_toy;
mod helpers;
mod traits;

pub use helpers::{helper_generate_random_gaussian_points, random_normal_points};
pub(crate) use traits::WrappedLogLikelihoodModel;
pub use traits::{
    LogLikelihoodModel, ParameterTransformer, ParameterTransformerConfig, Parametrised,
    ToyModelSamplerComponent,
};
