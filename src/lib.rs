//! essrs
//! An Ensemble Slice Sampler Built in Rust

/// Manages the storage of MCMC chain objects.
pub mod chains;

/// Main interface for running the MCMC program.
pub mod ess;

/// Auxillary functions.
/// - Standard likelihood functions such as normal distribution
/// - Functions to apply mathematical operations such as log-add-exp in an efficient manner
/// - Prior functions
/// - Search space transformation functions
pub mod functions;

/// Library of pre-built likelihood functions.
pub mod log_likelihood;

/// Library of available moves for the ensemble sampler.
pub mod moves;

/// Implementation of the randomised selection algorithms required by ensemble samplers
pub mod random;

/// State handler managing transition from i'th MCMC state to (i+1)'th state
pub mod state;

mod testing;
