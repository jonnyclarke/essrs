//! Numerical helper functions for use in constructing likelihood functions.
//!
//! * transforms: manipulating internal phase space to avoid hard boundary priors
//! * likelihood: common likelihood functions such as normal in 1/2 D
//! * priors: common priors available for use in likelihood functions
//! * utility: function wrappers to apply column wise to a 2d array

pub mod likelihood;
pub mod numerical;
pub mod priors;
pub mod transforms;
