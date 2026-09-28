//! This module contains helper functions implemented for ease of use

/// Module providing common likelihood functions such as normal distribution
pub mod likelihood;

/// Module providing numerical functions for use in custom log-likelihood functions. An example would be the log-add-exp algorithms used for numerically stable summation of log-likelihoods.
pub mod numerical;

// Module providing common functions used as priors.
pub mod priors;

// Module providing parameter transformation functions used for efficiently exploring parameter space.
pub mod transforms;
