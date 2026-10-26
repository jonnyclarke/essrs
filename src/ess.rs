//! Ensemble Slice Sampler
//!
//! This is the primary API to run the MCMC process.

mod builder;
mod class;
pub use builder::EnsembleSliceSamplerBuilder;
pub use class::EnsembleSliceSampler;
use thiserror::Error;

use crate::log_likelihood::LogLikelihoodError;

#[derive(Debug, Error)]
pub enum InitialisationError {
    #[error("initialisation of initial state has failed: {0}")]
    InvalidLogLikelihood(#[from] LogLikelihoodError),
}
